# Tile Exporter
Splits packed tilemaps into separate images unity can process.

# Preparing Tilemaps
Use pvrtextool to export tilemap png. Then use [Imagemagick](https://imagemagick.org/script/download.php) to convert to linear color space:
`magick input.png -colorspace sRGB -colorspace RGB output.png`. Then run the tilemap exporter.