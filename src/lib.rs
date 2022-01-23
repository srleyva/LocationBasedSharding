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
