from legacy_converters.core.stac import StacItem


def test_stac_item_values() -> None:
    geom = {
        "type": "Polygon",
        "coords": [[[0.0, 0.0], [0.5, 0.5], [1.0, 0.0], [0.5, -0.5], [0.0, 0.0]]],
    }
    bbox = [0.0, -0.5, 1.0, 0.5]
    item = StacItem(
        stac_extensions=[],
        id="item-id",
        bbox=bbox,
        geometry=geom,
        properties={},
        links=[],
        assets={},
    )

    assert item.type == "Feature"
    assert item.stac_version == "1.0.0"
    assert item.stac_extensions == []

    assert item.id == "item-id"
    assert item.bbox == bbox
    assert item.geometry == geom
    assert item.properties == {}
    assert item.links == {}
    assert item.assets == {}
