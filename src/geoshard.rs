use std::{collections::BTreeMap, sync::Arc};

use s2::{cap::Cap, cellid::CellID, point::Point, region::RegionCoverer, s1};
use serde_derive::{Deserialize, Serialize};

use crate::{
    cell_list::{CellList, CellScorer, UserCountScorer},
    users::{User, UserCollection},
    utils::ll,
};

pub const EARTH_RADIUS: f64 = 6.37e6f64;

pub struct GeoshardBuilder<Scorer> {
    storage_level: u64,
    users: UserCollection,
    cell_scorer: Scorer,
    min_shard_count: i32,
    max_shard_count: i32,
}

impl<Scorer> GeoshardBuilder<Scorer> {
    pub fn new(
        storage_level: u64,
        users: UserCollection,
        cell_scorer: Scorer,
        min_shard_count: i32,
        max_shard_count: i32,
    ) -> Self {
        Self {
            storage_level,
            cell_scorer,
            users,
            min_shard_count,
            max_shard_count,
        }
    }

    pub fn build(self) -> GeoShardCollection
    where
        Scorer: CellScorer,
    {
        self.generate_shards()
    }

    fn generate_shards(self) -> GeoShardCollection
    where
        Scorer: CellScorer,
    {
        // Calculate the score for each S2 cell based off of the provided Cell Scorer
        let cell_list = self
            .cell_scorer
            .score_cell_list(CellList::new(self.storage_level), self.users);
        let scored_cells = cell_list.scored_cells();

        // Get the total load in all the cells
        let total_load = scored_cells.iter().fold(0, |sum, i| sum + i.1);

        // Calculate the max_shard size and min_shard size based on shard count constraints
        let max_size = total_load / self.min_shard_count;
        let min_size = total_load / self.max_shard_count;

        let mut best_shards: Option<GeoShardCollection> = None;
        let mut min_standard_deviation = f64::MAX;

        // Try every possible shard size and return the one that has the lowest standard deviation
        for container_size in min_size..=max_size {
            let shards = GeoShardCollection::new(container_size, scored_cells);
            let standard_deviation = shards.standard_deviation();
            if standard_deviation < min_standard_deviation {
                min_standard_deviation = standard_deviation;
                best_shards = Some(shards);
            }
        }

        best_shards.unwrap()
    }
}

impl GeoshardBuilder<UserCountScorer> {
    pub fn user_count_scorer(
        storage_level: u64,
        users: UserCollection,
        min_shard_count: i32,
        max_shard_count: i32,
    ) -> Self {
        Self {
            storage_level,
            users,
            cell_scorer: UserCountScorer,
            max_shard_count,
            min_shard_count,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GeoShard {
    pub name: String,
    pub storage_level: u64,
    start: Option<String>,
    end: Option<String>,
    cell_count: i32,
    cell_score: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GeoShardCollection {
    #[serde(alias = "shards")]
    inner: Vec<GeoShard>,
}

impl TryFrom<&str> for GeoShardCollection {
    type Error = serde_json::Error;
    fn try_from(json_shards: &str) -> Result<Self, Self::Error> {
        serde_json::from_str(json_shards)
    }
}

impl GeoShardCollection {
    pub fn new(container_size: i32, scored_cells: &BTreeMap<CellID, i32>) -> Self {
        let first_cell = scored_cells.iter().next().unwrap();
        let mut shard = GeoShard {
            name: "geoshard_user_index_0".to_owned(),
            storage_level: first_cell.0.level(),
            start: Some(first_cell.0.to_token()),
            end: None,
            cell_count: 0,
            cell_score: 0,
        };
        let mut geo_shards = Vec::new();
        let mut geoshard_count = 1;
        for (cell_id, cell_score) in scored_cells {
            if shard.start == None {
                shard.start = Some(cell_id.to_token());
            }
            if shard.cell_score + cell_score < container_size {
                shard.cell_score += cell_score;
                shard.cell_count += 1;
            } else {
                shard.end = Some(cell_id.to_token());
                geo_shards.push(shard);
                shard = GeoShard {
                    name: format!("geoshard_user_index_{}", geoshard_count),
                    storage_level: cell_id.level(),
                    start: None,
                    end: None,
                    cell_count: 0,
                    cell_score: *cell_score,
                };
                geoshard_count += 1;
            }
        }
        if shard.cell_count != 0 {
            let last = scored_cells.iter().last().unwrap();
            shard.start = Some(last.0.to_token());
            shard.end = Some(last.0.to_token());
            shard.cell_count += 1;
            geo_shards.push(shard);
        }

        Self { inner: geo_shards }
    }

    pub fn standard_deviation(&self) -> f64 {
        let mean: f64 = self
            .inner
            .iter()
            .fold(0.0, |sum, x| sum + x.cell_score as f64)
            / self.inner.len() as f64;

        let varience: f64 = self
            .inner
            .iter()
            .map(|x| (x.cell_score as f64 - mean) * (x.cell_score as f64 - mean))
            .sum::<f64>()
            / self.inner.len() as f64;

        varience.sqrt()
    }
}

pub fn cell_id_from_long_lat(long: f64, lat: f64, storage_level: u64) -> CellID {
    let long_lat = ll!(long, lat);
    let cell_id = CellID::from(long_lat).parent(storage_level);
    cell_id
}

pub fn cell_ids_from_radius(long: f64, lat: f64, storage_level: u64, radius: u32) -> Vec<CellID> {
    let lon_lat = ll!(long, lat);

    let center_point = Point::from(lon_lat);

    let center_angle = s1::Deg(radius as f64 / EARTH_RADIUS).into();

    let cap = Cap::from_center_angle(&center_point, &center_angle);

    let region_cover = RegionCoverer {
        max_level: storage_level as u8,
        min_level: storage_level as u8,
        level_mod: 0,
        max_cells: 0,
    };
    region_cover.covering(&cap).0
}

#[derive(Debug, Clone)]
pub struct GeoShardSearcher {
    storage_level: u64,
    shards: Arc<GeoShardCollection>,
}

impl GeoShardSearcher {
    pub fn get_shard_for_user(&self, user: Box<dyn User>) -> &GeoShard {
        let location = user.location();
        let cell_id = CellID::from(location).parent(self.storage_level);
        self.get_shard_from_cell_id(cell_id)
    }

    pub fn get_shard_from_cell_id(&self, cell_id: CellID) -> &GeoShard {
        for geoshard in self.shards.inner.iter() {
            // Check if cell_id in shard
            let cell_id_start = CellID::from_token(geoshard.start.as_ref().unwrap().as_str());
            let cell_id_end = CellID::from_token(geoshard.end.as_ref().unwrap().as_str());
            if cell_id >= cell_id_start && cell_id <= cell_id_end {
                return &geoshard;
            }
        }
        self.shards.inner.last().unwrap()
    }

    pub fn get_shard_from_lng_lat(&self, lng: f64, lat: f64) -> &GeoShard {
        let cell_id = cell_id_from_long_lat(lng, lat, self.storage_level);
        self.get_shard_from_cell_id(cell_id)
    }

    pub fn get_shards_from_radius(&self, lng: f64, lat: f64, radius: u32) -> Vec<&GeoShard> {
        let mut geoshards = vec![];
        let cell_ids = cell_ids_from_radius(lng, lat, self.storage_level as u64, radius);
        for cell_id in cell_ids {
            geoshards.push(self.get_shard_from_cell_id(cell_id));
        }
        geoshards
    }
}

impl From<GeoShardCollection> for GeoShardSearcher {
    fn from(shards: GeoShardCollection) -> Self {
        let storage_level = shards.inner.first().unwrap().storage_level;
        let shards = Arc::new(shards);
        Self {
            storage_level,
            shards,
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;

    use rand::Rng;

    use s2::cellid::CellID;
    use s2::latlng::LatLng;
    use s2::s1;

    macro_rules! ll {
        ($lat:expr, $lng:expr) => {
            LatLng {
                lat: s1::Deg($lat).into(),
                lng: s1::Deg($lng).into(),
            }
        };
    }

    macro_rules! shard {
        ($cell_score:expr) => {
            GeoShard {
                name: "fake".to_owned(),
                storage_level: 0,
                start: None,
                end: None,
                cell_count: 0,
                cell_score: $cell_score,
            }
        };
    }

    pub struct RandomCellScore;

    #[test]
    fn test_shard_search() {
        let geoshard =
            GeoshardBuilder::user_count_scorer(4, Box::new(vec![].into_iter()), 40, 100).build();
        let geoshards = GeoShardSearcher::from(geoshard);

        let geoshard = geoshards.get_shard_from_lng_lat(34.181061, -103.345177);

        let cell_id = cell_id_from_long_lat(34.181061, -103.345177, 4);
        let cell_id_start = CellID::from_token(geoshard.start.as_ref().unwrap().as_str());
        let cell_id_end = CellID::from_token(geoshard.end.as_ref().unwrap().as_str());

        let range = cell_id_start..=cell_id_end;
        println!("Geoshard Range: {}-{}", cell_id_start, cell_id_end);
        println!("Geoshard cell: {}", cell_id);
        assert!(range.contains(&cell_id));
    }

    #[test]
    fn test_shard_radius_search() {
        let geoshard =
            GeoshardBuilder::new(4, Box::new(vec![].into_iter()), RandomCellScore, 40, 100).build();
        let geoshards = GeoShardSearcher::from(geoshard);
        let geoshards = geoshards.get_shards_from_radius(34.181061, -103.345177, 200);
        assert_eq!(geoshards.len(), 1);
    }

    #[test]
    fn test_generate_shards() {
        let geoshard =
            GeoshardBuilder::new(4, Box::new(vec![].into_iter()), RandomCellScore, 40, 100).build();

        let shards = geoshard.inner;

        if (shards.len() as i32) > 100 || (shards.len() as i32) < 40 {
            panic!("Shard len out of range: {}", shards.len());
        }
    }

    impl CellScorer for RandomCellScore {
        fn score_cell_list(&self, mut cell_list: CellList, _users: UserCollection) -> CellList {
            let mock_values = cell_list.mut_cell_list();
            let mut rng = rand::thread_rng();

            // Ocean
            for _ in 0..=1000 {
                let rand_lat = rng.gen_range(0.000000, 2000.000000);
                let rand_long = rng.gen_range(0.000000, 2000.000000);

                let cell_id = CellID::from(ll!(rand_lat, rand_long));
                let rand_load_count = rng.gen_range(0, 5);
                mock_values.insert(cell_id, rand_load_count);
            }

            // Small Cities
            for _ in 0..=100 {
                let rand_lat = rng.gen_range(0.000000, 2000.000000);
                let rand_long = rng.gen_range(0.000000, 2000.000000);

                let cell_id = CellID::from(ll!(rand_lat, rand_long));
                let rand_load_count = rng.gen_range(10, 100);
                mock_values.insert(cell_id, rand_load_count);
            }

            // Medium Cities
            for _ in 0..=50 {
                let rand_lat = rng.gen_range(0.000000, 2000.000000);
                let rand_long = rng.gen_range(0.000000, 2000.000000);

                let cell_id = CellID::from(ll!(rand_lat, rand_long));
                let rand_load_count = rng.gen_range(100, 500);
                mock_values.insert(cell_id, rand_load_count);
            }

            // Big Cities
            for _ in 0..=10 {
                let rand_lat = rng.gen_range(0.000000, 2000.000000);
                let rand_long = rng.gen_range(0.000000, 2000.000000);

                let cell_id = CellID::from(ll!(rand_lat, rand_long));
                let rand_load_count = rng.gen_range(1000, 2000);
                mock_values.insert(cell_id, rand_load_count);
            }

            cell_list
        }
    }

    #[test]
    fn test_standard_deviation() {
        let inner = vec![
            shard!(9),
            shard!(2),
            shard!(5),
            shard!(4),
            shard!(12),
            shard!(7),
            shard!(8),
            shard!(11),
            shard!(9),
            shard!(3),
            shard!(7),
            shard!(4),
            shard!(12),
            shard!(5),
            shard!(4),
            shard!(10),
            shard!(9),
            shard!(6),
            shard!(9),
            shard!(4),
        ];

        let geoshard_collection = GeoShardCollection { inner };

        let standard_dev = geoshard_collection.standard_deviation();
        assert_eq!(standard_dev, 2.9832867780352594_f64)
    }
}
