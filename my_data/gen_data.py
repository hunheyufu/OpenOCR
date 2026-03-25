"""Generate OpenOCR recognition data with arithmetic expressions.

Output format follows docs/finetune_rec.md:
- images in train/ and test/
- labels in rec_gt_train.txt and rec_gt_test.txt
- each label row: relative_image_path\ttext
"""

from __future__ import annotations

import argparse
import random
from fractions import Fraction
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


AUGMENT_PRESETS = {
    "safe": {
        "augment_prob": 0.5,
        "max_rotate_deg": 3.0,
        "max_translate_ratio": 0.03,
        "min_scale": 0.96,
        "max_scale": 1.04,
        "max_shear_deg": 2.0,
        "blur_prob": 0.1,
        "contrast_jitter": 0.1,
        "noise_prob": 0.12,
        "noise_std": 0.015,
        "shadow_prob": 0.1,
        "shadow_strength": 0.18,
        "shadow_blur": 8.0,
    },
    "mild": {
        "augment_prob": 0.75,
        "max_rotate_deg": 6.0,
        "max_translate_ratio": 0.06,
        "min_scale": 0.92,
        "max_scale": 1.08,
        "max_shear_deg": 4.0,
        "blur_prob": 0.2,
        "contrast_jitter": 0.2,
        "noise_prob": 0.2,
        "noise_std": 0.025,
        "shadow_prob": 0.18,
        "shadow_strength": 0.24,
        "shadow_blur": 10.0,
    },
    "medium": {
        "augment_prob": 0.9,
        "max_rotate_deg": 9.0,
        "max_translate_ratio": 0.08,
        "min_scale": 0.88,
        "max_scale": 1.12,
        "max_shear_deg": 6.0,
        "blur_prob": 0.3,
        "contrast_jitter": 0.25,
        "noise_prob": 0.3,
        "noise_std": 0.035,
        "shadow_prob": 0.28,
        "shadow_strength": 0.3,
        "shadow_blur": 12.0,
    },
    "strong": {
        "augment_prob": 1.0,
        "max_rotate_deg": 12.0,
        "max_translate_ratio": 0.1,
        "min_scale": 0.85,
        "max_scale": 1.15,
        "max_shear_deg": 8.0,
        "blur_prob": 0.35,
        "contrast_jitter": 0.3,
        "noise_prob": 0.4,
        "noise_std": 0.045,
        "shadow_prob": 0.38,
        "shadow_strength": 0.36,
        "shadow_blur": 14.0,
    },
}


def find_times_new_roman_font(font_path: str | None) -> Path:
    """Return a valid Times New Roman font path."""
    if font_path:
        path = Path(font_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Font not found: {font_path}")

    candidates = [
        Path("/usr/local/share/fonts/times-new-roman/Times New Roman.ttf"),
        Path("/mnt/c/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/times.ttf"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Times New Roman font not found. Pass --font-path explicitly, "
    )


def format_expression(numbers: Sequence[int], operators: Sequence[str]) -> str:
    display_map = {"*": "×", "/": "÷"}
    expr = str(numbers[0])
    for idx, op in enumerate(operators):
        expr += f"{display_map.get(op, op)}{numbers[idx + 1]}"
    return f"{expr}="


def format_parenthesized_expression(
    numbers: Sequence[int],
    operators: Sequence[str],
    group_left: bool,
) -> str:
    display_map = {"*": "×", "/": "÷"}
    if group_left:
        expr = (
            f"({numbers[0]}{display_map.get(operators[0], operators[0])}{numbers[1]})"
            f"{display_map.get(operators[1], operators[1])}{numbers[2]}"
        )
    else:
        expr = (
            f"{numbers[0]}{display_map.get(operators[0], operators[0])}"
            f"({numbers[1]}{display_map.get(operators[1], operators[1])}{numbers[2]})"
        )
    return f"{expr}="


def apply_operator(left: Fraction, right: Fraction, operator: str) -> Fraction | None:
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if right == 0:
        return None
    return left / right


def evaluate_expression(numbers: Sequence[int], operators: Sequence[str]) -> Fraction | None:
    values: List[Fraction] = [Fraction(n) for n in numbers]
    ops = list(operators)

    idx = 0
    while idx < len(ops):
        op = ops[idx]
        if op in ("*", "/"):
            left, right = values[idx], values[idx + 1]
            if op == "/":
                if right == 0:
                    return None
                merged = left / right
            else:
                merged = left * right
            values[idx : idx + 2] = [merged]
            ops.pop(idx)
            continue
        idx += 1

    result = values[0]
    for idx, op in enumerate(ops):
        right = values[idx + 1]
        if op == "+":
            result += right
        elif op == "-":
            result -= right
    return result


def build_binary_expression(min_value: int, max_value: int) -> str:
    op = random.choice(["+", "-", "*", "/"])

    # If value range only contains 0, division is impossible.
    if op == "/" and min_value == 0 and max_value == 0:
        op = "+"

    if op == "/":
        divisor = 0
        while divisor == 0:
            divisor = random.randint(min_value, max_value)
        quotient = random.randint(min_value, max_value)
        dividend = divisor * quotient
        left, right = dividend, divisor
    else:
        left = random.randint(min_value, max_value)
        right = random.randint(min_value, max_value)

    return format_expression([left, right], [op])


def build_expression(min_value: int, max_value: int, mixed_ratio: float) -> str:
    """Create arithmetic expression text that ends with '=' and has integer result."""
    use_mixed = random.random() < mixed_ratio
    if not use_mixed:
        return build_binary_expression(min_value=min_value, max_value=max_value)

    operators = ["+", "-", "*", "/"]
    for _ in range(1000):
        numbers = [
            random.randint(min_value, max_value),
            random.randint(min_value, max_value),
            random.randint(min_value, max_value),
        ]
        ops = [random.choice(operators), random.choice(operators)]

        use_parentheses = random.random() < 0.5
        if use_parentheses:
            group_left = random.random() < 0.5
            if group_left:
                grouped = apply_operator(Fraction(numbers[0]), Fraction(numbers[1]), ops[0])
                if grouped is None:
                    continue
                value = apply_operator(grouped, Fraction(numbers[2]), ops[1])
            else:
                grouped = apply_operator(Fraction(numbers[1]), Fraction(numbers[2]), ops[1])
                if grouped is None:
                    continue
                value = apply_operator(Fraction(numbers[0]), grouped, ops[0])

            if value is not None and value.denominator == 1:
                return format_parenthesized_expression(numbers, ops, group_left=group_left)

        value = evaluate_expression(numbers, ops)
        if value is not None and value.denominator == 1:
            return format_expression(numbers, ops)

    return build_binary_expression(min_value=min_value, max_value=max_value)


def render_text_image(
    text: str,
    font_path: Path,
    font_size: int,
    min_height: int,
    padding_x: int,
    padding_y: int,
    text_color: Tuple[int, int, int],
    background_color: Tuple[int, int, int],
) -> Image.Image:
    font = ImageFont.truetype(str(font_path), size=font_size)

    probe = Image.new("RGB", (10, 10), color=background_color)
    draw = ImageDraw.Draw(probe)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_width = int(right - left)
    text_height = int(bottom - top)

    width = text_width + 2 * padding_x
    height = max(min_height, text_height + 2 * padding_y)
    image = Image.new("RGB", (width, height), color=background_color)

    draw = ImageDraw.Draw(image)
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    draw.text((x, y), text, fill=text_color, font=font)
    return image


def apply_random_augmentation(
    image: Image.Image,
    max_rotate_deg: float,
    max_translate_ratio: float,
    min_scale: float,
    max_scale: float,
    max_shear_deg: float,
    blur_prob: float,
    contrast_jitter: float,
    noise_prob: float,
    noise_std: float,
    shadow_prob: float,
    shadow_strength: float,
    shadow_blur: float,
) -> Image.Image:
    """Apply light random transforms while keeping white background."""
    width, height = image.size

    # Add a safety border before transforms to avoid clipping text at edges.
    pad_ratio = 0.22 + max_translate_ratio
    pad = max(8, int(max(width, height) * pad_ratio))
    canvas_w = width + 2 * pad
    canvas_h = height + 2 * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    canvas.paste(image, (pad, pad))

    angle = random.uniform(-max_rotate_deg, max_rotate_deg)
    tx = int(width * random.uniform(-max_translate_ratio, max_translate_ratio))
    ty = int(height * random.uniform(-max_translate_ratio, max_translate_ratio))
    scale = random.uniform(min_scale, max_scale)
    shear_x = random.uniform(-max_shear_deg, max_shear_deg)

    aug_image = canvas.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255),
    )
    aug_image = aug_image.transform(
        (canvas_w, canvas_h),
        Image.Transform.AFFINE,
        data=(
            scale,
            shear_x / 100.0,
            tx,
            0.0,
            scale,
            ty,
        ),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255),
    )

    if random.random() < blur_prob:
        aug_image = aug_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))

    if contrast_jitter > 0:
        factor = random.uniform(1.0 - contrast_jitter, 1.0 + contrast_jitter)
        aug_image = ImageEnhance.Contrast(aug_image).enhance(factor)

    # Keep a clean reference for bbox detection so shadow/noise does not expand crop area.
    crop_reference = aug_image

    if noise_std > 0 and random.random() < noise_prob:
        noise_level = noise_std * 255.0
        base = np.asarray(aug_image.convert("RGB"), dtype=np.float32)
        noise = np.random.normal(0.0, noise_level, size=(base.shape[0], base.shape[1], 1))
        aug_image = Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8), mode="RGB")

    if shadow_strength > 0 and random.random() < shadow_prob:
        shadow_mask = Image.new("L", aug_image.size, 0)
        shadow_draw = ImageDraw.Draw(shadow_mask)
        shadow_w, shadow_h = aug_image.size

        # Use a large random ellipse to mimic uneven environmental shadow.
        ellipse_w = random.randint(max(12, int(shadow_w * 0.35)), max(13, int(shadow_w * 1.05)))
        ellipse_h = random.randint(max(12, int(shadow_h * 0.35)), max(13, int(shadow_h * 1.1)))
        center_x = random.randint(0, max(0, shadow_w - 1))
        center_y = random.randint(0, max(0, shadow_h - 1))
        left = center_x - ellipse_w // 2
        top = center_y - ellipse_h // 2
        right = left + ellipse_w
        bottom = top + ellipse_h
        opacity = int(255 * shadow_strength)
        shadow_draw.ellipse((left, top, right, bottom), fill=opacity)

        if shadow_blur > 0:
            shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
        aug_image = Image.composite(Image.new("RGB", aug_image.size, (0, 0, 0)), aug_image, shadow_mask)

    # Crop back to content region (with a small margin) to keep samples compact.
    gray = crop_reference.convert("L")
    content_mask = gray.point(lambda p: 255 if p < 245 else 0)
    bbox = content_mask.getbbox()
    if bbox is not None:
        margin = 4
        left = max(0, bbox[0] - margin)
        top = max(0, bbox[1] - margin)
        right = min(canvas_w, bbox[2] + margin)
        bottom = min(canvas_h, bbox[3] + margin)
        aug_image = aug_image.crop((left, top, right, bottom))

    return aug_image


def resolve_augmentation_params(args: argparse.Namespace) -> dict:
    params = dict(AUGMENT_PRESETS[args.augment_level])
    overrides = {
        "augment_prob": args.augment_prob,
        "max_rotate_deg": args.max_rotate_deg,
        "max_translate_ratio": args.max_translate_ratio,
        "min_scale": args.min_scale,
        "max_scale": args.max_scale,
        "max_shear_deg": args.max_shear_deg,
        "blur_prob": args.blur_prob,
        "contrast_jitter": args.contrast_jitter,
        "noise_prob": args.noise_prob,
        "noise_std": args.noise_std,
        "shadow_prob": args.shadow_prob,
        "shadow_strength": args.shadow_strength,
        "shadow_blur": args.shadow_blur,
    }
    for key, value in overrides.items():
        if value is not None:
            params[key] = value
    return params


def generate_split(
    split_name: str,
    count: int,
    output_dir: Path,
    label_file_path: Path,
    font_path: Path,
    font_size_range: Sequence[int],
    min_value: int,
    max_value: int,
    min_height: int,
    padding_x: int,
    padding_y: int,
    mixed_ratio: float,
    use_augmentation: bool,
    augmentation_prob: float,
    max_rotate_deg: float,
    max_translate_ratio: float,
    min_scale: float,
    max_scale: float,
    max_shear_deg: float,
    blur_prob: float,
    contrast_jitter: float,
    noise_prob: float,
    noise_std: float,
    shadow_prob: float,
    shadow_strength: float,
    shadow_blur: float,
) -> None:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    label_lines: List[str] = []
    for idx in range(count):
        text = build_expression(
            min_value=min_value,
            max_value=max_value,
            mixed_ratio=mixed_ratio,
        )
        font_size = random.randint(font_size_range[0], font_size_range[1])
        image = render_text_image(
            text=text,
            font_path=font_path,
            font_size=font_size,
            min_height=min_height,
            padding_x=padding_x,
            padding_y=padding_y,
            text_color=(0, 0, 0),
            background_color=(255, 255, 255),
        )
        if use_augmentation and random.random() < augmentation_prob:
            image = apply_random_augmentation(
                image=image,
                max_rotate_deg=max_rotate_deg,
                max_translate_ratio=max_translate_ratio,
                min_scale=min_scale,
                max_scale=max_scale,
                max_shear_deg=max_shear_deg,
                blur_prob=blur_prob,
                contrast_jitter=contrast_jitter,
                noise_prob=noise_prob,
                noise_std=noise_std,
                shadow_prob=shadow_prob,
                shadow_strength=shadow_strength,
                shadow_blur=shadow_blur,
            )

        file_name = f"{split_name}_{idx:07d}.png"
        image_path = split_dir / file_name
        image.save(image_path)
        label_lines.append(f"{split_name}/{file_name}\t{text}\n")

    label_file_path.write_text("".join(label_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate arithmetic OCR dataset for OpenOCR fine-tuning"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="dataset root directory (default: script directory)",
    )
    parser.add_argument("--train-count", type=int, default=5000)
    parser.add_argument("--test-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--font-path", default=None, help="Times New Roman .ttf path")
    parser.add_argument("--font-size-min", type=int, default=24)
    parser.add_argument("--font-size-max", type=int, default=42)
    parser.add_argument("--min-value", type=int, default=0)
    parser.add_argument("--max-value", type=int, default=99)
    parser.add_argument("--img-height", type=int, default=48)
    parser.add_argument("--padding-x", type=int, default=12)
    parser.add_argument("--padding-y", type=int, default=8)
    parser.add_argument(
        "--mixed-ratio",
        type=float,
        default=0.35,
        help="ratio of mixed expressions like a+b*c= or a÷b-c=",
    )
    parser.add_argument(
        "--augment-train",
        action="store_true",
        help="apply random image augmentation on train split",
    )
    parser.add_argument(
        "--augment-test",
        action="store_true",
        help="apply random image augmentation on test split",
    )
    parser.add_argument(
        "--augment-level",
        choices=["safe", "mild", "medium", "strong"],
        default="mild",
        help="preset for rotation/distortion strength",
    )
    parser.add_argument("--augment-prob", type=float, default=None)
    parser.add_argument("--max-rotate-deg", type=float, default=None)
    parser.add_argument("--max-translate-ratio", type=float, default=None)
    parser.add_argument("--min-scale", type=float, default=None)
    parser.add_argument("--max-scale", type=float, default=None)
    parser.add_argument("--max-shear-deg", type=float, default=None)
    parser.add_argument("--blur-prob", type=float, default=None)
    parser.add_argument("--contrast-jitter", type=float, default=None)
    parser.add_argument("--noise-prob", type=float, default=None)
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument("--shadow-prob", type=float, default=None)
    parser.add_argument("--shadow-strength", type=float, default=None)
    parser.add_argument("--shadow-blur", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.font_size_min > args.font_size_max:
        raise ValueError("--font-size-min must be <= --font-size-max")
    if args.min_value > args.max_value:
        raise ValueError("--min-value must be <= --max-value")
    if not 0 <= args.mixed_ratio <= 1:
        raise ValueError("--mixed-ratio must be in [0, 1]")
    aug_params = resolve_augmentation_params(args)
    if not 0 <= aug_params["augment_prob"] <= 1:
        raise ValueError("augment_prob must be in [0, 1]")
    if aug_params["min_scale"] <= 0 or aug_params["max_scale"] <= 0 or aug_params["min_scale"] > aug_params["max_scale"]:
        raise ValueError("scale range must be positive and min_scale <= max_scale")
    if not 0 <= aug_params["blur_prob"] <= 1:
        raise ValueError("blur_prob must be in [0, 1]")
    if aug_params["contrast_jitter"] < 0:
        raise ValueError("contrast_jitter must be >= 0")
    if not 0 <= aug_params["noise_prob"] <= 1:
        raise ValueError("noise_prob must be in [0, 1]")
    if aug_params["noise_std"] < 0:
        raise ValueError("noise_std must be >= 0")
    if not 0 <= aug_params["shadow_prob"] <= 1:
        raise ValueError("shadow_prob must be in [0, 1]")
    if not 0 <= aug_params["shadow_strength"] <= 1:
        raise ValueError("shadow_strength must be in [0, 1]")
    if aug_params["shadow_blur"] < 0:
        raise ValueError("shadow_blur must be >= 0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir) if args.output_dir else script_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    font_path = find_times_new_roman_font(args.font_path)

    generate_split(
        split_name="train",
        count=args.train_count,
        output_dir=output_dir,
        label_file_path=output_dir / "rec_gt_train.txt",
        font_path=font_path,
        font_size_range=(args.font_size_min, args.font_size_max),
        min_value=args.min_value,
        max_value=args.max_value,
        min_height=args.img_height,
        padding_x=args.padding_x,
        padding_y=args.padding_y,
        mixed_ratio=args.mixed_ratio,
        use_augmentation=args.augment_train,
        augmentation_prob=aug_params["augment_prob"],
        max_rotate_deg=aug_params["max_rotate_deg"],
        max_translate_ratio=aug_params["max_translate_ratio"],
        min_scale=aug_params["min_scale"],
        max_scale=aug_params["max_scale"],
        max_shear_deg=aug_params["max_shear_deg"],
        blur_prob=aug_params["blur_prob"],
        contrast_jitter=aug_params["contrast_jitter"],
        noise_prob=aug_params["noise_prob"],
        noise_std=aug_params["noise_std"],
        shadow_prob=aug_params["shadow_prob"],
        shadow_strength=aug_params["shadow_strength"],
        shadow_blur=aug_params["shadow_blur"],
    )

    generate_split(
        split_name="test",
        count=args.test_count,
        output_dir=output_dir,
        label_file_path=output_dir / "rec_gt_test.txt",
        font_path=font_path,
        font_size_range=(args.font_size_min, args.font_size_max),
        min_value=args.min_value,
        max_value=args.max_value,
        min_height=args.img_height,
        padding_x=args.padding_x,
        padding_y=args.padding_y,
        mixed_ratio=args.mixed_ratio,
        use_augmentation=args.augment_test,
        augmentation_prob=aug_params["augment_prob"],
        max_rotate_deg=aug_params["max_rotate_deg"],
        max_translate_ratio=aug_params["max_translate_ratio"],
        min_scale=aug_params["min_scale"],
        max_scale=aug_params["max_scale"],
        max_shear_deg=aug_params["max_shear_deg"],
        blur_prob=aug_params["blur_prob"],
        contrast_jitter=aug_params["contrast_jitter"],
        noise_prob=aug_params["noise_prob"],
        noise_std=aug_params["noise_std"],
        shadow_prob=aug_params["shadow_prob"],
        shadow_strength=aug_params["shadow_strength"],
        shadow_blur=aug_params["shadow_blur"],
    )

    print(f"Done. Dataset generated under: {output_dir.resolve()}")
    print(f"Font: {font_path}")
    if args.augment_train or args.augment_test:
        print(f"Augmentation preset: {args.augment_level}, params: {aug_params}")


if __name__ == "__main__":
    main()

