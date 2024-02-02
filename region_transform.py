import re


def convert_region_file(input_file, output_file):
    with open(input_file, 'r') as input_file:
        content = input_file.read()

    # Extracting point coordinates using regex
    points = re.findall(r'point\(([\d.]+),([\d.]+)\)', content)

    converted_content = "# Region file format: DS9 version 4.1\n"
    converted_content += ("global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 "
                          "highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
    converted_content += "image\n"

    for i, (x, y) in enumerate(points, start=0):
        if i == 0:
            target = "NG0552-4001_TIC-21725734_S41Z"
            converted_content += f"annulus({x},{y}, 9.0,14.0)  # text={{{target}}}\n"
        else:
            converted_content += f"annulus({x},{y}, 20.0,30.0)  # text={{{i}}}\n"

    with open(output_file, 'w') as output_file:
        output_file.write(converted_content)


path = '/Users/u5500483/Downloads/DATA_MAC/CMOS/'
# Example usage
convert_region_file(path + 'ds9.reg', path + 'output.reg')


