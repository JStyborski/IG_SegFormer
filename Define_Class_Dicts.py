import csv

def define_class_dicts(filePath):

    # Read the CSV file with classes and colors (RGB)
    with open(filePath, 'r') as f:
        csvReader = csv.reader(f)
        next(csvReader) # Skip header row
        rows = []
        for row in csvReader:
            rows.append(row)

    # Parse through list and define id/label, id/color, and label/id dictionaries
    id2label = {}
    id2color = {}
    for idx, row in enumerate(rows):
        id2label[idx] = row[0]
        id2color[idx] = [int(row[1]), int(row[2]), int(row[3])]

    label2id = {v: k for k, v in id2label.items()}
    color2id = {tuple(v): k for k, v in id2color.items()}

    return id2label, label2id, id2color, color2id

