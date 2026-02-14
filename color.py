#Basic color theory & algorithms (concise)
#Here are the top 3 specific topics to learn 
#from your color theory summary:

#CIE Color Spaces & Perceptual Uniformity
#Learn about CIE XYZ, CIE Lab, and CIE Luv color spaces, 
#which are designed for accurate color measurement and 
#device-independent color representation.

#Additive Color Theory (RGB) & Color Mixing Algorithms
#Understand how RGB channels mix light, the math behind 
#hue, saturation, value, and how algorithms like linear 
#interpolation and color temperature gradients work.

#Color Appearance Models & Accessibility
#Study models like CIECAM02 for context-aware color prediction, 
# advanced color difference formulas (CIEDE2000), and accessibility 
# standards (WCAG contrast, color blindness simulation).

# Key scientists:
# - Isaac Newton: Color spectrum, color wheel (1704)
# - Thomas Young & Hermann von Helmholtz: Trichromatic theory (1802, 1850s)
# - James Clerk Maxwell: Color matching, additive color (1855)
# - Albert H. Munsell: Color system (1905)
# - Richard S. Hunter: L,a,b color space (1948)
# - W. David Wright & John Guild: CIE color matching (1920s-1931)
#
# Modern color theory & algorithms:
# - CIE color spaces (CIE XYZ, CIE Lab, CIE Luv) for perceptual uniformity and device independence
# - sRGB and linear RGB for digital imaging and displays
# - Color appearance models (CIECAM02, CAM16) for context-aware color prediction
# - Advanced color difference formulas (CIEDE2000)
# - Algorithms for color management, gamut mapping, and ICC profiles
# - Machine learning for colorization, segmentation, and palette generation
# - Accessibility: WCAG contrast, color blindness simulation
#
#
# Theory (short):
# - Additive color (RGB): light mixes by adding channels; black is (0,0,0) and white is (255,255,255).
# - Hue is the angle on a color wheel; saturation is colorfulness; value is brightness.
# - Complementary colors are 180 deg apart on the hue wheel; triadic colors are 120 deg apart.
# - Linear interpolation models additive mixing; perceptual brightness is approximated by relative luminance.
# - Contrast ratio compares luminance to estimate readability between two colors.
from typing import Tuple

RGB = Tuple[int, int, int]

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """RGB (0-255) -> HSV (H in degrees, S,V in 0-1)."""
    r_, g_, b_ = r/255.0, g/255.0, b/255.0
    cmax, cmin = max(r_, g_, b_), min(r_, g_, b_)
    delta = cmax - cmin

    # Hue
    if delta == 0:
        h = 0.0
    elif cmax == r_:
        h = (60 * ((g_ - b_) / delta) + 360) % 360
    elif cmax == g_:
        h = (60 * ((b_ - r_) / delta) + 120) % 360
    else:
        h = (60 * ((r_ - g_) / delta) + 240) % 360

    # Saturation
    s = 0.0 if cmax == 0 else delta / cmax
    v = cmax
    return h, s, v

def hsv_to_rgb(h: float, s: float, v: float) -> RGB:
    """HSV -> RGB (0-255)."""
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if 0 <= h < 60:
        rp, gp, bp = c, x, 0
    elif 60 <= h < 120:
        rp, gp, bp = x, c, 0
    elif 120 <= h < 180:
        rp, gp, bp = 0, c, x
    elif 180 <= h < 240:
        rp, gp, bp = 0, x, c
    elif 240 <= h < 300:
        rp, gp, bp = x, 0, c
    else:
        rp, gp, bp = c, 0, x
    r, g, b = (rp + m), (gp + m), (bp + m)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))

def complementary_color(rgb: RGB) -> RGB:
    """Complementary color by rotating hue 180 deg."""
    h, s, v = rgb_to_hsv(*rgb)
    h2 = (h + 180) % 360
    return hsv_to_rgb(h2, s, v)

def triadic_colors(rgb: RGB) -> Tuple[RGB, RGB]:
    """Triadic colors: hue +120 deg and +240 deg."""
    h, s, v = rgb_to_hsv(*rgb)
    return (hsv_to_rgb((h + 120) % 360, s, v),
            hsv_to_rgb((h + 240) % 360, s, v))

def mix_colors(rgb1: RGB, rgb2: RGB, t: float) -> RGB:
    """Linear interpolation (additive mixing) between two colors."""
    t = clamp01(t)
    return (int(round(rgb1[0] * (1 - t) + rgb2[0] * t)),
            int(round(rgb1[1] * (1 - t) + rgb2[1] * t)),
            int(round(rgb1[2] * (1 - t) + rgb2[2] * t)))

def relative_luminance(rgb: RGB) -> float:
    """WCAG relative luminance using linearized sRGB."""
    def linearize(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = map(linearize, rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def contrast_ratio(rgb1: RGB, rgb2: RGB) -> float:
    """WCAG contrast ratio."""
    l1 = relative_luminance(rgb1)
    l2 = relative_luminance(rgb2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)

def color_temperature(t: float) -> RGB:
    """Simple warm-to-cool gradient (0=warm, 1=cool)."""
    warm = (255, 160, 64)  # orange
    cool = (64, 160, 255)  # blue
    return mix_colors(warm, cool, clamp01(t))

if __name__ == "__main__":
    base = (120, 80, 200)
    print("Base RGB:", base)
    print("HSV:", rgb_to_hsv(*base))

    print("Complement:", complementary_color(base))
    print("Triadic:", triadic_colors(base))

    print("Mix 25% toward white:", mix_colors(base, (255, 255, 255), 0.25))
    print("Luminance:", round(relative_luminance(base), 4))
    print("Contrast vs white:", round(contrast_ratio(base, (255, 255, 255)), 2))

    for t in [0.0, 0.5, 1.0]:
        print(f"Temp {t}: {color_temperature(t)}")