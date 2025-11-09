from rembg_demo import remove
from rembg.session_factory import new_session
from PIL import Image


class RembgSession:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.session = None if model_name == 'none' else new_session(
            model_name)

    def set_model(self, model_name: str):
        if model_name != self.model_name:
            self.model_name = model_name
            self.session = None if model_name == 'none' else new_session(
                model_name)

    def remove_bg(self, image: Image.Image, **kwargs) -> Image.Image:
        image = image.convert("RGB")
        print(f"Using model: {self.model_name} {self.session}")
        return remove(image, session=self.session)
