# Tile Exporter
Splits packed tilemap pngs into separate images unity can process.

# Preparing Tilemaps
   1. Use [PvrTexTool](https://developer.imaginationtech.com/pvrtextool/) to export 
      tilemap png. Make sure to use the `File -> Save Image` option to save the texture 
      to png or the exporter won't work
   2. Use [Imagemagick](https://imagemagick.org/script/download.php) to convert to 
      linear color space:
      `magick input.png -colorspace sRGB -colorspace RGB output.png`
   3. Verify directories at the top of the TileExporter script, then run it
