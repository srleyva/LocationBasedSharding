use s2::latlng::LatLng;

pub type UserCollection = Box<dyn Iterator<Item = Box<dyn User>>>;

pub trait User {
    fn location(&self) -> LatLng;
}
