from PIL import Image

from omnibrain.app.services.vision.chart_preview import (
    verify_bbox,
    verify_image_quality,
    resolve_image_path,
    build_chart_preview,
    build_chart_preview_list,
)


def test_verify_bbox_valid():
    assert verify_bbox(
        [10, 20, 100, 100],
        image_width=340,
        image_height=452,
    ) is True


def test_verify_bbox_invalid():
    assert verify_bbox(
        [-1, 20, 100, 100],
        image_width=340,
        image_height=452,
    ) is False

    assert verify_bbox(
        [300, 400, 100, 100],
        image_width=340,
        image_height=452,
    ) is False


def test_verify_image_quality(tmp_path):
    image_path = tmp_path / "chart.png"

    Image.new("RGB", (340, 452)).save(image_path)

    result = verify_image_quality(str(image_path))

    assert result["valid"] is True
    assert result["width"] == 340
    assert result["height"] == 452


def test_resolve_image_path():
    original = "output/images/page_2_image_1.jpeg"
    cropped = "output/images/page_2_image_1_crop.png"

    assert resolve_image_path(original, cropped) == cropped
    assert resolve_image_path(original, None) == original


def test_build_chart_preview(tmp_path):
    image_path = tmp_path / "chart.png"

    Image.new("RGB", (340, 452)).save(image_path)

    metadata = {
        "page_number": 2,
        "image_path": str(image_path),
        "dimensions": (340, 452),
        "bbox": [10, 20, 100, 100],
        "caption": "Fig. 1. Test Chart",
    }

    vlm_result = {
        "chart_type": "bar",
        "title": "Test Chart",
    }

    preview = build_chart_preview(
        image_metadata=metadata,
        vlm_result=vlm_result,
    )

    assert preview["page_number"] == 2
    assert preview["image_path"] == str(image_path)
    assert preview["bbox"] == [10, 20, 100, 100]
    assert preview["bbox_valid"] is True
    assert preview["resolution"]["valid"] is True
    assert preview["vlm"]["chart_type"] == "bar"


def test_build_chart_preview_list(tmp_path):
    image1 = tmp_path / "chart1.png"
    image2 = tmp_path / "chart2.png"

    Image.new("RGB", (340, 452)).save(image1)
    Image.new("RGB", (500, 500)).save(image2)

    extracted_images = [
        {
            "page_number": 2,
            "image_path": str(image1),
            "dimensions": (340, 452),
            "bbox": [10, 20, 100, 100],
            "caption": "Fig. 1",
        },
        {
            "page_number": 3,
            "image_path": str(image2),
            "dimensions": (500, 500),
            "bbox": [20, 30, 200, 200],
            "caption": "Fig. 2",
        },
    ]

    vlm_results = [
        {"chart_type": "bar", "title": "Chart 1"},
        {"chart_type": "line", "title": "Chart 2"},
    ]

    cropped_paths = [
        str(tmp_path / "chart1_crop.png"),
        str(tmp_path / "chart2_crop.png"),
    ]

    previews = build_chart_preview_list(
        extracted_images=extracted_images,
        vlm_results=vlm_results,
        cropped_image_paths=cropped_paths,
    )

    assert len(previews) == 2

    assert previews[0]["image_path"] == cropped_paths[0]
    assert previews[0]["vlm"]["chart_type"] == "bar"

    assert previews[1]["image_path"] == cropped_paths[1]
    assert previews[1]["vlm"]["chart_type"] == "line"