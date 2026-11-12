// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Dfx

open System
open System.IO
open System.Runtime.InteropServices
open SkiaSharp


/// Contains auto-opened utilities related to the DiffSharp programming model.
[<AutoOpen>]
module ImageUtil =

    // Pixels are processed as RGBA half floats. These encode to 16-bit PNG files and keep the precision of 16-bit sources.
    let private floatImageInfo (width:int) (height:int) (alphaType:SKAlphaType) =
        SKImageInfo(width, height, SKColorType.RgbaF16, alphaType)

    let private toHalfBits (v:float32) = BitConverter.HalfToInt16Bits(Half.op_Explicit(Math.Clamp(v, 0.0f, 1.0f)))

    /// Returns the size of a resized image. A zero width or height is computed from the other, preserving the aspect ratio.
    let private resizeTarget (sourceWidth:int) (sourceHeight:int) (width:int, height:int) =
        if width < 0 || height < 0 then failwithf "Expecting width (%A) and height (%A) >= 0" width height
        if width = 0 && height = 0 then failwithf "Expecting width (%A) or height (%A) > 0" width height
        let scale (size:int) (target:int) (source:int) = max 1 (int (Math.Round(float size * float target / float source)))
        if width = 0 then scale sourceWidth height sourceHeight, height
        elif height = 0 then width, scale sourceHeight width sourceWidth
        else width, height

    /// Returns the bitmap resized with cubic interpolation, disposing the original, or the bitmap itself if no resize is given.
    let private resizeBitmap (bitmap:SKBitmap) (resize:option<int*int>) =
        match resize with
        | None -> bitmap
        | Some size ->
            let width, height = resizeTarget bitmap.Width bitmap.Height size
            let resized = bitmap.Resize(floatImageInfo width height bitmap.AlphaType, SKFilterQuality.High)
            bitmap.Dispose()
            if isNull resized then failwithf "Cannot resize image to %Ax%A" width height
            resized

    /// Returns true if the file is a PNG image with 16 bits per channel.
    let private isPng16 (fileName:string) =
        use stream = File.OpenRead(fileName)
        let header = Array.zeroCreate<byte> 26
        stream.ReadAtLeast(header, header.Length, false) = header.Length
        && header[0..7] = [|137uy; 80uy; 78uy; 71uy; 13uy; 10uy; 26uy; 10uy|]
        && header[24] = 16uy

    /// Saves the given pixel array to a file and optionally resizes it in the process. Resizing uses bicubic interpolation. Supports .png and .jpg formats.
    let saveImage (pixels:float32[,,]) (fileName:string) (resize:option<int*int>) =
        let format =
            if fileName.EndsWith(".jpg") then SKEncodedImageFormat.Jpeg
            elif fileName.EndsWith(".png") then SKEncodedImageFormat.Png
            else failwithf "Expecting fileName (%A) to end with .png or .jpg" fileName
        let c, h, w = pixels.GetLength(0), pixels.GetLength(1), pixels.GetLength(2)
        let data = Array.zeroCreate<int16> (w * h * 4)
        for y=0 to h-1 do
            for x=0 to w-1 do
                let r, g, b =
                    if c = 1 then
                        let v = pixels[0, y, x]
                        v, v, v
                    else
                        pixels[0, y, x], pixels[1, y, x], pixels[2, y, x]
                let i = (y * w + x) * 4
                data[i] <- toHalfBits r
                data[i + 1] <- toHalfBits g
                data[i + 2] <- toHalfBits b
                data[i + 3] <- toHalfBits 1.0f
        let bitmap = new SKBitmap(floatImageInfo w h SKAlphaType.Opaque)
        Marshal.Copy(data, 0, bitmap.GetPixels(), data.Length)
        use bitmap = resizeBitmap bitmap resize
        // The quality setting applies to JPEG only
        use encoded = bitmap.Encode(format, 75)
        if isNull encoded then failwithf "Cannot encode image %A" fileName
        use stream = File.Create(fileName)
        encoded.SaveTo(stream)

    /// Loads a pixel array from a file and optionally resizes it in the process. Resizing uses bicubic interpolation.
    let loadImage (fileName:string) (resize:option<int*int>) =
        use codec = SKCodec.Create(fileName)
        if isNull codec then failwithf "Cannot load image %A: the file format is not supported" fileName
        // 8-bit images are read exactly; 16-bit PNG images are read as half floats to keep their extra precision
        let colorType = if isPng16 fileName then SKColorType.RgbaF16 else SKColorType.Rgba8888
        let bitmap = new SKBitmap(SKImageInfo(codec.Info.Width, codec.Info.Height, colorType, SKAlphaType.Unpremul))
        let result = codec.GetPixels(bitmap.Info, bitmap.GetPixels())
        if result <> SKCodecResult.Success then
            bitmap.Dispose()
            failwithf "Cannot load image %A: %A" fileName result
        use bitmap = resizeBitmap bitmap resize
        let w, h = bitmap.Width, bitmap.Height
        let channel : int -> float32 =
            if bitmap.ColorType = SKColorType.RgbaF16 then
                let data = Array.zeroCreate<int16> (w * h * 4)
                Marshal.Copy(bitmap.GetPixels(), data, 0, data.Length)
                fun i -> float32 (BitConverter.Int16BitsToHalf(data[i]))
            else
                let data = Array.zeroCreate<byte> (w * h * 4)
                Marshal.Copy(bitmap.GetPixels(), data, 0, data.Length)
                fun i -> float32 data[i] / 255.0f
        Array3D.init 3 h w (fun c y x -> channel ((y * w + x) * 4 + c))


[<AutoOpen>]
module ImageExtensions =
    type Tensor with
        /// <summary>Save tensor to an image file using png or jpg format</summary>
        member t.saveImage(fileName:string, ?pixelMin:double, ?pixelMax:double, ?normalize:bool, ?resize:int*int, ?gridCols:int) =
            let pixels:Tensor = t.move(Device.CPU).toImage(?pixelMin=pixelMin, ?pixelMax=pixelMax, ?normalize=normalize, ?gridCols=gridCols)
            saveImage (pixels.float32().toArray() :?> float32[,,]) fileName resize

        /// <summary>Load an image file and return it as a tensor</summary>
        static member loadImage(fileName:string, ?normalize:bool, ?resize:int*int, ?device: Device, ?dtype: Dtype, ?backend: Backend) =
            let normalize = defaultArg normalize false
            let pixels = loadImage fileName resize
            let pixels:Tensor = Tensor.create(pixels, ?device=device, ?dtype=dtype, ?backend=backend)
            if normalize then pixels.normalize() else pixels


    type dfx with
        /// <summary>Load an image file as a tensor.</summary>
        /// <param name="fileName">The file name of the image to load.</param>
        /// <param name="normalize">If True, shift the image to the range (0, 1).</param>
        /// <param name="resize">An optional new size for the image.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member loadImage(fileName:string, ?normalize:bool, ?resize:int*int, ?device: Device, ?dtype: Dtype, ?backend: Backend) =
            Tensor.loadImage(fileName=fileName, ?normalize=normalize, ?resize=resize, ?device=device, ?dtype=dtype, ?backend=backend)

        /// <summary>Save a given Tensor into an image file.</summary>
        /// <remarks>If the input tensor has 4 dimensions, then make a single image grid.</remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="fileName">The name of the file to save to.</param>
        /// <param name="pixelMin">The minimum pixel value.</param>
        /// <param name="pixelMax">The maximum pixel value.</param>
        /// <param name="normalize">If True, shift the image to the range (0, 1), by the min and max values specified by range.</param>
        /// <param name="resize">An optional new size for the image.</param>
        /// <param name="gridCols">Number of columns of images in the grid.</param>
        static member saveImage(input:Tensor, fileName:string, ?pixelMin:double, ?pixelMax:double, ?normalize:bool, ?resize:int*int, ?gridCols:int) =
            input.saveImage(fileName=fileName, ?pixelMin=pixelMin, ?pixelMax=pixelMax, ?normalize=normalize, ?resize=resize, ?gridCols=gridCols)
