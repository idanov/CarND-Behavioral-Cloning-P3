import csv


def parse_record(base_dir, row):
    angle = 0
    for i in range(0, 3):
        if len(row[i].strip()) > 0:
            row[i] = base_dir + row[i].strip()
        angle = float(row[3].strip())
    return [angle, row[0], row[1], row[2]]


def load_csv(base_dir, filename):
    with open(base_dir + filename, 'rt') as f:
        reader = csv.reader(f)
        arr = [parse_record(base_dir, row) for row in reader]
        return arr


def split_data(csv_rows, threshold):
    go_straight = [row for row in csv_rows if abs(row[0]) <= threshold]
    go_left = [row for row in csv_rows if row[0] < -threshold]
    go_right = [row for row in csv_rows if row[0] > threshold]
    return go_straight, go_left, go_right


def strip_side_cam_images(csv_rows):
    return [[row[0], row[1]] for row in csv_rows]


def add_side_cam_images(csv_rows, side_cam_offset):
    left_images = [[min(row[0] + side_cam_offset, 1), row[2]] for row in csv_rows]
    right_images = [[max(row[0] - side_cam_offset, -1), row[3]] for row in csv_rows]
    center_images = strip_side_cam_images(csv_rows)
    return left_images + right_images + center_images
