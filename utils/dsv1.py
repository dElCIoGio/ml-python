import struct
import numpy as np

MAGIC = b"DSV1"
VERSION = 1

DTYPE_CODE = {
    np.uint8: 1,
    np.float32: 2,
    np.int32: 2,  # careful: reusing 2 would be ambiguous; better define unique codes (see note below)
}

# Better: unique codes
DTYPE_CODE = {
    np.uint8: 1,
    np.float32: 2,
    np.int32: 3,
}

def write_dsv1(path: str, images: np.ndarray, labels: np.ndarray) -> None:
    """
    images: (N,H,W,C) uint8 or float32
    labels: (N,) uint8 or int32
    """
    if images.ndim != 4:
        raise ValueError("images must be (N,H,W,C)")
    if labels.ndim != 1:
        raise ValueError("labels must be (N,)")

    N, H, W, C = images.shape
    if labels.shape[0] != N:
        raise ValueError("labels length must match images N")

    img_dtype = images.dtype.type
    lbl_dtype = labels.dtype.type
    if img_dtype not in DTYPE_CODE or lbl_dtype not in DTYPE_CODE:
        raise ValueError(f"Unsupported dtypes: {images.dtype}, {labels.dtype}")

    # Ensure contiguous raw bytes
    images_c = np.ascontiguousarray(images)
    labels_c = np.ascontiguousarray(labels)

    header = struct.pack(
        "<4sIIIIIII8s",   # little-endian
        MAGIC,
        VERSION,
        N, H, W, C,
        DTYPE_CODE[img_dtype],
        DTYPE_CODE[lbl_dtype],
        b"\x00" * 8
    )

    with open(path, "wb") as f:
        f.write(header)
        f.write(images_c.tobytes(order="C"))
        f.write(labels_c.tobytes(order="C"))



CODE_DTYPE = {
    1: np.uint8,
    2: np.float32,
    3: np.int32,
}

def read_dsv1(path: str):
    with open(path, "rb") as f:
        header_bytes = f.read(struct.calcsize("<4sIIIIIII8s"))
        magic, version, N, H, W, C, img_code, lbl_code, _ = struct.unpack("<4sIIIIIII8s", header_bytes)

        if magic != b"DSV1":
            raise ValueError("Not a DSV1 file")
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")

        img_dtype = CODE_DTYPE[img_code]
        lbl_dtype = CODE_DTYPE[lbl_code]

        img_count = N * H * W * C
        images = np.frombuffer(f.read(img_count * np.dtype(img_dtype).itemsize), dtype=img_dtype)
        images = images.reshape((N, H, W, C))

        labels = np.frombuffer(f.read(N * np.dtype(lbl_dtype).itemsize), dtype=lbl_dtype)

        return images, labels