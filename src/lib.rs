pub mod cell_list;
pub mod geoshard;
pub mod users;

pub mod utils {
    macro_rules! ll {
        ($lng:expr, $lat:expr) => {
            s2::latlng::LatLng {
                lat: s2::s1::Deg($lat).into(),
                lng: s2::s1::Deg($lng).into(),
            }
        };
    }

    pub(crate) use ll;
}

#[cfg(test)]
mod test {
    use std::{collections::HashMap, fs::File, io::Write};

    use crate::geoshard::{test::FakeUser, GeoshardBuilder, GeoshardSearcher};

    #[test]
    fn test_geoshard_searcher() {
        let users: Vec<FakeUser> = (0..2000).map(|_| FakeUser::new()).collect();

        let geoshards = GeoshardBuilder::user_count_scorer(8, users.iter(), 40, 100).build();
        let searcher = GeoshardSearcher::from(geoshards);

        let user_database = users.iter().fold(HashMap::new(), |mut database, user| {
            let cell_id = searcher.get_shard_for_user(user);
            database
                .entry(cell_id.name())
                .or_insert(Vec::new())
                .push(user.clone());
            database
        });

        for user in users.iter() {
            let cell_id = searcher.get_shard_for_user(user);
            let shard_collection = user_database
                .get(cell_id.name())
                .expect("cell_id not found in database");

            assert!(shard_collection.contains(user));
        }

        let shards = searcher.shards();

        // TODO: Implement serde serialze and deserialze
        // let json_shards = serde_json::to_string(shards).unwrap();
        // let mut shard_file = File::create("shard.json").expect("could not create shard file");
        // shard_file
        //     .write_all(&json_shards.as_bytes())
        //     .expect("could not write json shards");
    }

    #[test]
    fn test_geoshard_properties() {
        let users: Vec<FakeUser> = (0..2000).map(|_| FakeUser::new()).collect();

        let geoshards = GeoshardBuilder::user_count_scorer(8, users.iter(), 40, 100).build();

        assert_eq!(
            geoshards
                .shards()
                .iter()
                .map(|shard| shard.cell_count())
                .sum::<usize>(),
            393217
        );
    }
}
