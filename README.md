# Location Based Sharding Algorithm

Based off [Tinder own location based sharding algorithm](https://medium.com/tinder-engineering/geosharded-recommendations-part-1-sharding-approach-d5d54e0ec77a), the algorithm will create shards based off of the "heat" of users in a particular location. This flexible sharding schema allows less dense locations to be encompassed in less shards and more densely populated areas to be split in to different shards.

# Sharding metadata

When sharding is initiated, it creates shards based off the current user score. These shards need to be persisted in some sort of persistent store or database. This can be used for something like elastic search or location based sharding in any kind of database. 

Some trade offs for consideration:

- Users in dense cities may have to hit multiple shards
- uses who live on edges may wander across and their information will have to moved across

# Example

```rust
// This can also be serialized and pulled from Redis
// let geoshards = GeoshardBuilder::from(&json_string_from_ddb)).unwrap();
let geoshards = GeoshardBuilder::user_count_scorer(8, Box::new(vec![].into_iter()), 40, 100).build();
let shard_searcher = GeoShardSearcher::from(geoshards);
let shard_user_is_in = shard_searcher.get_shard_user(some_user);
// Query you index based off the shard ^^^
```