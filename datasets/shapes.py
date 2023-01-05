import math
import random
from typing import Optional

from PIL import Image, ImageDraw
from dataclasses import dataclass


@dataclass()
class Shape:
    margin: int = 5

    def intersects(self, shape: 'Shape'):
        a_1, a_2 = self.bounding_rect()
        b_1, b_2 = shape.bounding_rect()

        return a_1[0] < b_2[0] and a_2[0] > b_1[0] and a_1[1] < b_2[1] and a_2[1] > b_1[1]

    def bounding_rect(self) -> ((int, int), (int, int)):
        raise NotImplementedError

    def random_init(self, height, width, scale):
        raise NotImplementedError

    def center(self) -> (int, int):
        raise NotImplementedError


@dataclass()
class Triangle(Shape):
    p1: (int, int) = None
    p2: (int, int) = None
    p3: (int, int) = None

    def bounding_rect(self):
        top_left = min(self.p1[0], self.p2[0], self.p3[0]) - self.margin, min(self.p1[1], self.p2[1],
                                                                              self.p3[1]) - self.margin

        bottom_right = max(self.p1[0], self.p2[0], self.p3[0]) + self.margin, max(self.p1[1], self.p2[1],
                                                                                  self.p3[1]) + self.margin
        return top_left, bottom_right

    def random_init(self, height, width, scale):
        left, top = self.margin, self.margin
        right, bottom = width - left, height - top
        self.p1 = random.randint(left, right - scale), random.randint(top, bottom - scale)
        radius = scale // 2 + random.random() * (scale - scale // 2)
        p2_x_offset = random.random() * radius
        p2_y_offset = (radius ** 2 - p2_x_offset ** 2) ** 0.5
        self.p2 = self.p1[0] + p2_x_offset, self.p1[1] + p2_y_offset
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        angle = math.pi / 3
        self.p3 = self.p1[0] + math.cos(angle) * dx + math.sin(angle) * dy, self.p1[1] + math.sin(
            -angle) * dx + math.cos(angle) * dy

    def center(self) -> (int, int):
        return (self.p1[0] + self.p2[0] + self.p3[0]) / 3, (self.p1[1] + self.p2[1] + self.p3[1]) / 3


@dataclass()
class Square(Shape):
    top_left: (int, int) = None
    bottom_right: (int, int) = None

    def bounding_rect(self):
        return (self.top_left[0] - self.margin, self.top_left[1] - self.margin), (
            self.bottom_right[0] + self.margin, self.bottom_right[1] + self.margin
        )

    def random_init(self, height, width, scale):
        left, top = self.margin, self.margin
        right, bottom = width - left, height - top

        self.top_left = random.randint(left, right - scale), random.randint(top, bottom - scale)
        size = (scale // 3) + int(random.random() * (scale - scale // 3))
        self.bottom_right = self.top_left[0] + size, self.top_left[1] + size

    def center(self) -> (int, int):
        return (self.bottom_right[0] + self.top_left[0]) / 2, (self.bottom_right[1] + self.top_left[1]) / 2


@dataclass()
class Circle(Shape):
    center_point: (int, int) = None
    radius: int = None

    def bounding_rect(self) -> ((int, int), (int, int)):
        return (self.center_point[0] - self.radius - self.margin, self.center_point[1] - self.radius - self.margin), (
            self.center_point[0] + self.radius + self.margin, self.center_point[1] + self.radius + self.margin
        )

    def random_init(self, height, width, scale):
        left, top = self.margin, self.margin
        right, bottom = width - left, height - top
        self.center_point = random.randint(left + scale, right - scale), random.randint(top + scale, bottom - scale)
        self.radius = scale // 3 + int(random.random() * (scale - scale // 3))

    def center(self) -> (int, int):
        return self.center_point


def scatter_shapes(shape_types, height, width, scale, num_tries=10000) -> list[Shape]:
    shapes: list[Shape] = []
    for shape_type in shape_types:
        new_shape: Shape = shape_type()
        counter = num_tries
        success = False
        while not success:
            success = True
            new_shape.random_init(height, width, scale)
            for shape in shapes:
                if new_shape.intersects(shape):
                    counter -= 1
                    success = False
                    break
            if counter == 0:
                raise RuntimeError("Failed to generate a non-intersecting shape")
        shapes.append(new_shape)
    assert len(shapes) == len(shape_types)
    return shapes


@dataclass()
class DatasetEntry:
    shapes: list[Shape]
    image: Image.Image
    mask: Optional[Image.Image]


def generate_image(num_shapes: int, create_mask: bool, size: (int, int) = (224, 224),
                   background=(255, 255, 255)) -> DatasetEntry:
    chosen_shape_types = random.choices([Circle, Triangle, Square], k=num_shapes)
    chosen_colors = random.choices(['red', 'green', 'blue'], k=num_shapes)
    shapes = scatter_shapes(chosen_shape_types, size[0], size[1], 40)

    image = Image.new('RGB', size, background)
    draw = ImageDraw.Draw(image)
    mask, mask_draw = None, None
    if create_mask:
        mask = Image.new('1', size)
        mask_draw = ImageDraw.Draw(mask)

    for shape, color in zip(shapes, chosen_colors):
        if isinstance(shape, Triangle):
            draw.polygon((shape.p1, shape.p2, shape.p3), fill=color)
            if create_mask:
                mask_draw.polygon((shape.p1, shape.p2, shape.p3), fill='white')
        elif isinstance(shape, Circle):
            draw.ellipse((shape.center_point[0] - shape.radius, shape.center_point[1] - shape.radius,
                          shape.center_point[0] + shape.radius, shape.center_point[1] + shape.radius), fill=color)
            if create_mask:
                mask_draw.ellipse((shape.center_point[0] - shape.radius, shape.center_point[1] - shape.radius,
                                   shape.center_point[0] + shape.radius, shape.center_point[1] + shape.radius),
                                  fill='white')
        elif isinstance(shape, Square):
            draw.rectangle((shape.top_left, shape.bottom_right), fill=color)
            if create_mask:
                mask_draw.rectangle((shape.top_left, shape.bottom_right), fill='white')
    return DatasetEntry(shapes, image, mask)


entry = generate_image(8, True)
entry.image.save("shapes/test.png", "PNG")
entry.mask.save("shapes/test_mask.png", "PNG")
