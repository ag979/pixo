import httpx

from tests.integration.config import BASE_URL, TEST_IMAGE_PATH, DEBUG_IMAGE_PATH
from app.client import retrieve_image, segment_image, upload_image


def test_end_to_end_workflow():

    with httpx.Client(timeout=httpx.Timeout(20.0)) as client:

        image_id = upload_image(client, BASE_URL, TEST_IMAGE_PATH)

        segment_image(client, BASE_URL, image_id)

        retrieve_image(client, BASE_URL, DEBUG_IMAGE_PATH)
