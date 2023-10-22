from bruhanimate.bruhscreen import Screen
from bruhanimate.bruhrenderer import *
import bruhanimate.images as images

def boot(screen):
    image = images.text_to_image("F N B B\nD E V S", padding_top_bottom=1, padding_left_right=3)
    renderer = FocusRenderer(screen, 400, 0, image, "plasma", " ", transparent=False, start_frame=30, reverse=True, start_reverse=220)
    renderer.effect.update_color_properties(color=True, characters=True, random_color=True)
    renderer.effect.update_grey_scale_size(10)
    renderer.effect.update_plasma_values(10, 26, 19, 41)
    renderer.run(end_message=True)

if __name__ == "__main__":
    Screen.show(boot)
    