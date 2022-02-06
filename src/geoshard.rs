#![deny(missing_docs)]
//! This Module is used to build GeoshardCollections and the searchers aroung them
//!
//! The way this logic work, S2 library breaks the globe down into set cells with variable granularity based on level
//! these cells are then scored in the `build` function using the provided Scorer trait (See UserCountScore for an implementation example)
//! the cells are then distributed into the shard configuration with the least standard deviation between shards
//!
//! # Examples
//!
//! ```rust
//! use location_based_sharding::geoshard::{GeoshardBuilder, GeoshardSearcher};
//! #[cfg(test)]
//! use location_based_sharding::geoshard::test::FakeUser;
//!
//! #[cfg(test)]
//! let geoshards = GeoshardBuilder::user_count_scorer(8, Box::new(vec![].into_iter()), 40, 100).build();
//! #[cfg(test)]
//! let shard_searcher = GeoshardSearcher::from(geoshards);
//! // let shard_user_is_in = shard_searcher.get_shard_user(some_user);
//! ```

use std::collections::BTreeMap;

use s2::{
    cap::Cap, cellid::CellID, cellunion::CellUnion, latlng::LatLng, point::Point,
    region::RegionCoverer, s1,
};
use serde::{ser::SerializeStruct, Deserialize, Serialize};
use serde_derive::{Deserialize, Serialize};

use crate::{
    cell_list::{CellList, CellScorer, UserCountScorer},
    users::User,
};

const EARTH_RADIUS: f64 = 6.37e6f64;

/// The `GeoshardBuilder<Scorer>` type. This used to generate and score shards baed on provided Scorer.
/// Generating Shards can potentially be an expensive operation, which is why the builder pattern is
/// used, so that consumers can explictly decide when to generate the shards.
///
/// Used to generate S2 Cells, Score them, and then organize them into right sized shards. The basis for most constructs here
///
/// # Examples
///
/// Create a GeoshardCollection with the Score implicitly set
///
/// ```rust
/// use location_based_sharding::geoshard::GeoshardBuilder;
///
/// #[cfg(test)]
/// use location_based_sharding::geoshard::test::FakeUser;
///
/// #[cfg(test)]
/// let geoshards = GeoshardBuilder::user_count_scorer(4, Box::new(vec![FakeUser::new()].into_iter()), 40, 100).build();
/// ```
pub struct GeoshardBuilder<Scorer, UserCollection> {
    storage_level: u64,
    users: UserCollection,
    cell_scorer: Scorer,
    min_shard_count: i32,
    max_shard_count: i32,
}

impl<Scorer, UserCollection> GeoshardBuilder<Scorer, UserCollection> {
    /// Constructs a new Builder for Geoshards
    ///
    /// `storage_level` is a S2 specific configuration. It determines the granularity of cells
    /// NOTE: higher level does indicate more granular cell which takes a lot more compute to
    /// process and shard
    /// https://s2geometry.io/resources/s2cell_statistics.html
    ///
    /// `users` is the collection of users to iterate over when scoring, this type is an iterator of type User, as such an iterator
    /// can be implemented for any given type (such as paging from DDB, to in-memory vec) See Cell Score Type
    ///
    /// `cell_scorer` is the algorithm for scoring S2 cells, this can be a heristic on the user (See CellScoreer trait)
    ///
    /// `min_shard_count` is the minimum number of shards that can be in your system
    ///
    /// `max_shard_count` is the max number of shards in the system
    ///
    /// # Examples
    ///
    /// Creating a GeoshardBuilder with UserCountScorer explicityl set
    ///
    /// ```rust
    /// use location_based_sharding::{cell_list::UserCountScorer, geoshard::GeoshardBuilder};
    ///
    /// #[cfg(test)]
    /// use location_based_sharding::geoshard::test::FakeUser;
    ///
    /// #[cfg(test)]
    /// let geoshards = GeoshardBuilder::new(4, Box::new(vec![FakeUser::new()].into_iter()), UserCountScorer, 40, 100).build();
    /// ```
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

    /// `build` will actually build the S2 CellList from the given storage level, score each cell, and
    /// then generate shards for every possible shard count and find the one with the lowest standard
    /// deviation between them.
    pub fn build<T>(self) -> GeoshardCollection
    where
        Scorer: CellScorer<UserCollection>,
        UserCollection: Iterator<Item = T>,
        T: User,
    {
        // Calculate the score for each S2 cell based off of the provided Cell Scorer
        let cell_list = self
            .cell_scorer
            .score_cell_list(CellList::new(self.storage_level), self.users);
        let scored_cells = cell_list.cell_list();

        // Get the total load in all the cells
        let total_load = scored_cells.iter().fold(0, |sum, i| sum + i.1);

        // Calculate the max_shard size and min_shard size based on shard count constraints
        let max_size = total_load / self.min_shard_count;
        let min_size = total_load / self.max_shard_count;

        let mut best_shards: Option<GeoshardCollection> = None;
        let mut min_standard_deviation = f64::MAX;

        // Try every possible shard size and return the one that has the lowest standard deviation
        for container_size in min_size..=max_size {
            let shards = GeoshardCollection::new(container_size, scored_cells, self.storage_level);
            let standard_deviation = shards.standard_deviation();
            if standard_deviation < min_standard_deviation {
                min_standard_deviation = standard_deviation;
                best_shards = Some(shards);
            }
        }

        best_shards.unwrap()
    }
}

impl<UserCollection> GeoshardBuilder<UserCountScorer, UserCollection> {
    /// Create a `GeoshardBuilder<UserCountScorer>` where a cells given scorer is defaultly set
    /// to score based off UserCount in that area
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

/// `Geoshard` represents one shard...each shard contains a variable amount of cells
#[derive(Debug)]
pub struct Geoshard {
    name: String,
    storage_level: u64,
    start: CellID,
    end: CellID,
    cell_score: i32,
    cell_union: CellUnion,
    size: usize,
}

impl Serialize for Geoshard {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("Geoshard", 5)?;
        state.serialize_field("name", &self.name)?;
        state.serialize_field("storage_level", &self.storage_level)?;
        state.serialize_field("start", &self.start.to_token())?;
        state.serialize_field("end", &self.end.to_token())?;
        state.serialize_field("cell_score", &self.cell_score)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Geoshard {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        todo!()
    }
}

impl Geoshard {
    /// returns a new geoshard
    pub fn new(
        name: String,
        start: CellID,
        end: CellID,
        cell_score: i32,
        storage_level: u64,
        size: usize,
    ) -> Self {
        let cell_union = CellUnion::from_range(start, end);
        Self {
            size,
            start,
            end,
            name,
            storage_level,
            cell_score,
            cell_union,
        }
    }

    /// name returns the name of the shard
    pub fn name(&self) -> &str {
        &self.name
    }

    /// cell_count returns the cell_count for this geoshard
    pub fn cell_count(&self) -> usize {
        self.size
    }

    /// returns the starting cell
    pub fn start(&self) -> &CellID {
        &self.start
    }

    /// returns the end cell
    pub fn end(&self) -> &CellID {
        &self.end
    }

    /// Returns a cell union from this shard
    pub fn cell_union(&self) -> &CellUnion {
        &self.cell_union
    }

    /// returns the stroage level of the cells in this shard
    pub fn storage_level(&self) -> u64 {
        self.storage_level
    }
}

/// `GeoshardCollection` is the collection of shards generated by by the builder
#[derive(Debug)]
pub struct GeoshardCollection {
    storage_level: u64,
    shards: Vec<Geoshard>,
}

impl GeoshardCollection {
    /// returns shards in this collection
    pub fn shards(&self) -> &Vec<Geoshard> {
        &self.shards
    }
}

// impl TryFrom<&str> for GeoshardCollection {
//     type Error = serde_json::Error;
//     fn try_from(json_shards: &str) -> Result<Self, Self::Error> {
//         serde_json::from_str(json_shards)
//     }
// }

impl GeoshardCollection {
    /// Constructs a new `GeoshardCollection`
    ///
    /// this will actually iterate over each s2 cell and assign it a shard
    /// taking into account the limit of shards allowed in the system
    pub fn new(
        container_size: i32,
        scored_cells: &BTreeMap<CellID, i32>,
        storage_level: u64,
    ) -> Self {
        let mut current_start = scored_cells.iter().next().unwrap().0;
        let mut current_end = scored_cells.iter().next().unwrap().0;
        let mut current_cell_count = 0;
        let mut current_score = 0;

        let mut shards = Vec::new();
        let mut geoshard_count = 1;

        for (cell_id, cell_score) in scored_cells {
            if cell_score + current_score > container_size {
                let shard = Geoshard::new(
                    format!("geoshard_user_index_{}", geoshard_count),
                    *current_start,
                    *current_end,
                    current_score,
                    cell_id.level(),
                    current_cell_count,
                );
                shards.push(shard);
                current_start = cell_id;
                current_cell_count = 0;
                current_score = 0;
                geoshard_count += 1;
            }
            current_end = cell_id;
            current_cell_count += 1;
            current_score += cell_score;
        }

        if geoshard_count != shards.len() {
            let shard = Geoshard::new(
                format!("geoshard_user_index_{}", geoshard_count),
                *current_start,
                *current_end,
                current_score,
                storage_level,
                current_cell_count,
            );

            shards.push(shard);
        }

        Self {
            shards,
            storage_level,
        }
    }

    /// Calculates the standard deviation between shards
    pub fn standard_deviation(&self) -> f64 {
        let mean: f64 = self
            .shards
            .iter()
            .fold(0.0, |sum, x| sum + x.cell_score as f64)
            / self.shards.len() as f64;

        let varience: f64 = self
            .shards
            .iter()
            .map(|x| (x.cell_score as f64 - mean) * (x.cell_score as f64 - mean))
            .sum::<f64>()
            / self.shards.len() as f64;

        varience.sqrt()
    }
}

/// `GeoshardSearcher` actual contains logic to find a users given shard, given a user
#[derive(Debug)]
pub struct GeoshardSearcher {
    storage_level: u64,
    shards: GeoshardCollection,
}

impl GeoshardSearcher {
    /// return shards
    pub fn shards(&self) -> &GeoshardCollection {
        &self.shards
    }

    /// returns shard for given user
    pub fn get_shard_for_user<T>(&self, user: T) -> &Geoshard
    where
        T: User,
    {
        let location = user.location();
        self.get_shard_from_location(location)
    }

    /// returns the given `CellID` for given location
    pub fn get_cell_id_from_location(&self, location: &LatLng) -> CellID {
        CellID::from(location).parent(self.storage_level)
    }

    /// returns shard from given location
    pub fn get_shard_from_location(&self, location: &LatLng) -> &Geoshard {
        self.get_shard_from_cell_id(&self.get_cell_id_from_location(location))
    }

    /// returns a shard for given cell ID
    pub fn get_shard_from_cell_id(&self, cell_id: &CellID) -> &Geoshard {
        for geoshard in self.shards.shards.iter() {
            if geoshard.cell_union().contains_cellid(cell_id) {
                return geoshard;
            }
        }
        self.shards.shards.last().unwrap()
    }

    /// returns the given shard in a location and radius
    pub fn get_shards_from_radius(&self, location: &LatLng, radius: u32) -> Vec<&Geoshard> {
        self.cell_ids_from_radius(location, radius)
            .into_iter()
            .map(|cell_id| self.get_shard_from_cell_id(&cell_id))
            .collect()
    }

    /// Gives all the CellIDs in a given radius in miles
    pub fn cell_ids_from_radius(&self, location: &LatLng, radius: u32) -> Vec<CellID> {
        let center_point = Point::from(location);

        let center_angle = s1::Deg(radius as f64 / EARTH_RADIUS).into();

        let cap = Cap::from_center_angle(&center_point, &center_angle);

        let region_cover = RegionCoverer {
            max_level: self.storage_level as u8,
            min_level: self.storage_level as u8,
            level_mod: 0,
            max_cells: 0,
        };
        region_cover.covering(&cap).0
    }
}

impl From<GeoshardCollection> for GeoshardSearcher {
    fn from(shards: GeoshardCollection) -> Self {
        let storage_level = shards.storage_level;
        Self {
            storage_level,
            shards,
        }
    }
}

#[cfg(test)]
pub mod test {

    use super::*;
    use crate::utils::ll;

    use rand::Rng;

    use lazy_static::lazy_static;
    use rand::{distributions::Alphanumeric, prelude::SliceRandom, thread_rng};
    use s2::cellid::CellID;

    struct RandCityFactory {
        cities: Vec<LatLng>,
    }
    impl RandCityFactory {
        fn new_city(&self) -> LatLng {
            let mut rng = rand::thread_rng();
            self.cities.choose(&mut rng).unwrap().clone()
        }

        fn cities(&self) -> &Vec<LatLng> {
            &self.cities
        }
    }

    impl Default for RandCityFactory {
        fn default() -> Self {
            let cities: Vec<LatLng> = vec![
                ll!(40.745255, 40.745255),
                ll!(34.155834, 34.155834),
                ll!(42.933334, 42.933334),
                ll!(42.095554, 42.095554),
                ll!(38.846668, 38.846668),
                ll!(41.392502, 41.392502),
                ll!(27.192223, 27.192223),
                ll!(31.442778, 31.442778),
                ll!(40.560001, 40.560001),
                ll!(33.193611, 33.193611),
                ll!(41.676388, 41.676388),
                ll!(41.543056, 41.543056),
                ll!(39.554443, 39.554443),
                ll!(44.513332, 44.513332),
                ll!(37.554169, 37.554169),
                ll!(32.349998, 32.349998),
                ll!(29.499722, 29.499722),
                ll!(33.038334, 33.038334),
                ll!(43.614166, 43.614166),
                ll!(41.55611, 41.55611),
                ll!(34.00, 34.00),
                ll!(26.709723, 26.709723),
                ll!(38.005001, 38.005001),
                ll!(35.970554, 35.970554),
                ll!(25.942122, 25.942122),
                ll!(33.569443, 33.569443),
                ll!(39.799999, 39.799999),
                ll!(34.073334, 34.073334),
                ll!(40.606388, 40.606388),
                ll!(30.601389, 30.601389),
                ll!(38.257778, 38.257778),
                ll!(37.977222, 37.977222),
                ll!(42.373611, 42.373611),
                ll!(32.965557, 32.965557),
                ll!(37.871666, 37.871666),
                ll!(38.951561, 38.951561),
                ll!(33.950001, 33.950001),
                ll!(30.216667, 30.216667),
                ll!(42.580276, 42.580276),
                ll!(36.316666, 36.316666),
                ll!(37.034946, 37.034946),
                ll!(40.689167, 40.689167),
                ll!(33.630554, 33.630554),
                ll!(39.903057, 39.903057),
                ll!(25.978889, 25.978889),
                ll!(35.846111, 35.846111),
                ll!(34.156113, 34.156113),
                ll!(41.18639, 41.18639),
                ll!(40.914745, 40.914745),
                ll!(42.259445, 42.259445),
                ll!(41.520557, 41.520557),
                ll!(33.124722, 33.124722),
                ll!(39.106667, 39.106667),
                ll!(42.101391, 42.101391),
                ll!(37.210388, 37.210388),
                ll!(33.866669, 33.866669),
                ll!(26.012501, 26.012501),
                ll!(38.438332, 38.438332),
                ll!(33.211666, 33.211666),
                ll!(37.070831, 37.070831),
                ll!(43.536388, 43.536388),
                ll!(45.633331, 45.633331),
                ll!(42.271389, 42.271389),
                ll!(30.455, 30.455),
                ll!(32.492222, 32.492222),
                ll!(33.466667, 33.466667),
                ll!(32.361668, 32.361668),
                ll!(41.763889, 41.763889),
                ll!(35.199165, 35.199165),
                ll!(37.661388, 37.661388),
                ll!(32.907223, 32.907223),
                ll!(33.669445, 33.669445),
                ll!(39.710835, 39.710835),
                ll!(32.705002, 32.705002),
                ll!(39.099724, 39.099724),
                ll!(35.1175, 35.1175),
                ll!(39.791, 39.791),
                ll!(39.983334, 39.983334),
                ll!(30.266666, 30.266666),
                ll!(32.779167, 32.779167),
                ll!(37.487846, 37.487846),
                ll!(35.25528, 35.25528),
                ll!(29.700001, 29.700001),
                ll!(26.838619, 26.838619),
                ll!(38.473625, 38.473625),
                ll!(29.749907, 29.749907),
                ll!(40.191891, 40.191891),
                ll!(33.830517, 33.830517),
                ll!(34.496212, 34.496212),
                ll!(37.54129, 37.54129),
                ll!(36.082157, 36.082157),
                ll!(32.698437, 32.698437),
                ll!(33.580944, 33.580944),
                ll!(33.427204, 33.427204),
                ll!(34.028622, 34.028622),
                ll!(32.609856, 32.609856),
                ll!(33.405746, 33.405746),
                ll!(34.603817, 34.603817),
                ll!(44.840797, 44.840797),
                ll!(71.290558, 71.290558),
            ];
            Self { cities }
        }
    }

    lazy_static! {
        static ref RANDOM_CITY_FACTORY: RandCityFactory = RandCityFactory::default();
    }

    #[derive(Clone)]
    pub struct FakeUser {
        pub name: String,
        location: LatLng,
    }

    impl PartialEq for FakeUser {
        fn eq(&self, other: &Self) -> bool {
            other.name == self.name
        }
    }

    impl FakeUser {
        pub fn new() -> Self {
            let name: String = thread_rng()
                .sample_iter(&Alphanumeric)
                .take(30)
                .map(char::from)
                .collect();
            Self {
                name,
                location: RANDOM_CITY_FACTORY.new_city(),
            }
        }
    }

    impl User for &FakeUser {
        fn location(&self) -> &LatLng {
            &self.location
        }
    }

    macro_rules! shard {
        ($cell_score:expr) => {
            Geoshard::new(
                "fake-shard".to_owned(),
                CellID::from_token("00001"),
                CellID::from_token("00003"),
                $cell_score,
                0,
                2,
            )
        };
    }

    pub struct RandomCellScore;

    #[test]
    fn test_shard_search() {
        let geoshards =
            GeoshardBuilder::user_count_scorer(4, Box::new(vec![FakeUser::new()].iter()), 40, 100)
                .build();
        let geoshard_searcher = GeoshardSearcher::from(geoshards);

        let geoshard = geoshard_searcher.get_shard_from_location(&ll!(34.181061, -103.345177));

        let cell_id = geoshard_searcher.get_cell_id_from_location(&ll!(34.181061, -103.345177));

        assert!(geoshard.cell_union().contains_cellid(&cell_id));
    }

    #[test]
    fn test_shard_radius_search() {
        let geoshard = GeoshardBuilder::new(
            4,
            Box::new(vec![FakeUser::new()].iter()),
            RandomCellScore,
            40,
            100,
        )
        .build();
        let geoshards = GeoshardSearcher::from(geoshard);
        let geoshards = geoshards.get_shards_from_radius(&ll!(34.181061, -103.345177), 200);
        assert_eq!(geoshards.len(), 1);
    }

    #[test]
    fn test_generate_shards() {
        let geoshard = GeoshardBuilder::new(
            4,
            Box::new(vec![FakeUser::new()].iter()),
            RandomCellScore,
            40,
            100,
        )
        .build();

        let shards = geoshard.shards;

        if (shards.len() as i32) > 100 || (shards.len() as i32) < 40 {
            panic!("Shard len out of range: {}", shards.len());
        }
    }

    impl<UserCollection> CellScorer<UserCollection> for RandomCellScore {
        fn score_cell_list<T>(&self, mut cell_list: CellList, _users: UserCollection) -> CellList {
            let mock_values = cell_list.mut_cell_list();
            let mut rng = rand::thread_rng();

            // Ocean
            for _ in 0..=1000 {
                let rand_lat = rng.gen_range(0.000000..2000.000000);
                let rand_long = rng.gen_range(0.000000..2000.000000);

                let cell_id = CellID::from(ll!(rand_lat, rand_long));
                let rand_load_count = rng.gen_range(0..5);
                mock_values.insert(cell_id, rand_load_count);
            }

            // Small Cities
            for _ in 0..=100 {
                let rand_lat = rng.gen_range(0.000000..2000.000000);
                let rand_long = rng.gen_range(0.000000..2000.000000);

                let cell_id = CellID::from(ll!(rand_lat, rand_long));
                let rand_load_count = rng.gen_range(10..100);
                mock_values.insert(cell_id, rand_load_count);
            }

            // Medium Cities
            for _ in 0..=50 {
                let rand_lat = rng.gen_range(0.000000..2000.000000);
                let rand_long = rng.gen_range(0.000000..2000.000000);

                let cell_id = CellID::from(ll!(rand_lat, rand_long));
                let rand_load_count = rng.gen_range(100..500);
                mock_values.insert(cell_id, rand_load_count);
            }

            // Big Cities
            for _ in 0..=10 {
                let rand_lat = rng.gen_range(0.000000..2000.000000);
                let rand_long = rng.gen_range(0.000000..2000.000000);

                let cell_id = CellID::from(ll!(rand_lat, rand_long));
                let rand_load_count = rng.gen_range(1000..2000);
                mock_values.insert(cell_id, rand_load_count);
            }

            cell_list
        }
    }

    #[test]
    fn test_standard_deviation() {
        let shards = vec![
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

        let geoshard_collection = GeoshardCollection {
            shards,
            storage_level: 4,
        };

        let standard_dev = geoshard_collection.standard_deviation();
        assert_eq!(standard_dev, 2.9832867780352594_f64)
    }
}
