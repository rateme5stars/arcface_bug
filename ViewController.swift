//
//  ViewController.swift
//  test_arcface
//
//  Created by Dzung Ngo on 17/05/2023.
//

import UIKit

class ViewController: UIViewController {
    let leoImage: UIImageView = {
        let imageView = UIImageView()
        imageView.image = UIImage(named: "0.jpg")
        return imageView
    }()
    
    let dzungImage: UIImageView = {
        let imageView = UIImageView()
        imageView.image = UIImage(named: "1.jpg")
        return imageView
    }()
    
    let label: UILabel = {
        let label = UILabel()
        label.textColor = .black
        return label
    }()

    private var arcfaceHandler: ArcFaceHandler? = ArcFaceHandler (
        modelFileInfo: ArcFace.modelInfo
    )

    override func viewDidLoad() {
        view.backgroundColor = .systemBlue
        leoImage.frame = CGRect(x: 60, y: 100, width: 112, height: 112)
        dzungImage.frame = CGRect(x: 250, y: 100, width: 112, height: 112)
        label.frame = CGRect(x: 100, y: 300, width: 300, height: 20)
        view.addSubview(leoImage)
        view.addSubview(dzungImage)
        view.addSubview(label)

        // Convert UIImage to CVPixelBuffer
        guard let leoCgImage = leoImage.image?.cgImage else {return} // leo image
        guard let dzungCgImage = dzungImage.image?.cgImage else {return} // dzung image
        
//        let leoPixelBuffer = convertUIImageToCVPixelBuffer(image: leoImage.image!)
//        let dzungPixelBuffer = convertUIImageToCVPixelBuffer(image: dzungImage.image!)
        
        let leoPixelBuffer = convertCGImageToCVPixelBuffer(cgImage: leoCgImage)
        let dzungPixelBuffer = convertCGImageToCVPixelBuffer(cgImage: dzungCgImage)
        
        guard let leoEmbedding = self.arcfaceHandler?.runModel(frame: leoPixelBuffer!) else { return }
        guard let dzungEmbedding = self.arcfaceHandler?.runModel(frame: dzungPixelBuffer!) else { return }

        let score = calCosineSimilarity(personEmbedding: leoEmbedding, realTimeEmbedding: dzungEmbedding)
        label.text = "leo vs dzung (scaled): \(score)"
        super.viewDidLoad()
    }
    
    private func calCosineSimilarity(personEmbedding a: [Float], realTimeEmbedding b: [Float]) -> Float {
        guard a.count == b.count else {
            fatalError("The two arrays must have the same number of elements")
        }
        
        var dotProduct: Float = 0.0
        var aMagnitude: Float = 0.0
        var bMagnitude: Float = 0.0

        for i in 0..<a.count {
            dotProduct += a[i] * b[i]
            aMagnitude += a[i] * a[i]
            bMagnitude += b[i] * b[i]
        }

        aMagnitude = sqrt(aMagnitude)
        bMagnitude = sqrt(bMagnitude)

        let cosineSimilarity = dotProduct / (aMagnitude * bMagnitude)
        let scaledCosineSimilarity = (cosineSimilarity + 1) / 2
        return scaledCosineSimilarity
    }
    
    func convertCGImageToCVPixelBuffer(cgImage: CGImage) -> CVPixelBuffer? {
        let width = cgImage.width
        let height = cgImage.height
        let pixelFormat = kCVPixelFormatType_32BGRA

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(nil, width, height, pixelFormat, nil, &pixelBuffer)
        guard status == kCVReturnSuccess, let unwrappedPixelBuffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(unwrappedPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(unwrappedPixelBuffer)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = CVPixelBufferGetBytesPerRow(unwrappedPixelBuffer)
        let context = CGContext(data: pixelData, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue)
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        CVPixelBufferUnlockBaseAddress(unwrappedPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

        return unwrappedPixelBuffer
    }

//    func convertUIImageToCVPixelBuffer(image: UIImage) -> CVPixelBuffer? {
//        let width = Int(image.size.width)
//        let height = Int(image.size.height)
//
//        let attributes: [String: Any] = [
//            kCVPixelBufferCGImageCompatibilityKey as String: kCFBooleanTrue!,
//            kCVPixelBufferCGBitmapContextCompatibilityKey as String: kCFBooleanTrue!,
//            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
//        ]
//
//        var pixelBuffer: CVPixelBuffer?
//        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attributes as CFDictionary, &pixelBuffer)
//
//        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
//            return nil
//        }
//
//        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
//        let pixelData = CVPixelBufferGetBaseAddress(buffer)
//
//        let colorSpace = CGColorSpaceCreateDeviceRGB()
//        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
//        let context = CGContext(data: pixelData, width: width, height: height, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: colorSpace, bitmapInfo: bitmapInfo.rawValue)
//
//        context?.draw(image.cgImage!, in: CGRect(x: 0, y: 0, width: width, height: height))
//
//        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
//
//        return pixelBuffer
//    }
}

