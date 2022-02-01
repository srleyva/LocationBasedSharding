#![deny(missing_docs)]
//! User related things, such as the Collection defintion
//! and User trait
use s2::latlng::LatLng;

/// User is the trait for a given user that needs to be distributed
/// all that is required is a location in the format thats required
/// by S2 to find the correct cell
pub trait User {
    /// location returns the S2 LatLng that is used to find the given cell_id
    fn location(&self) -> &LatLng;
}
