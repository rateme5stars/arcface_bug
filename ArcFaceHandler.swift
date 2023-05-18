//
//  ArcFaceHandler.swift
//  Runner
//
//  Created by Dzung Ngo on 25/04/2023.
//


import CoreImage
import TensorFlowLite
import UIKit

//typealias arcfaceFileInfo = (name: String, extension: String)

typealias fileInfo = (name: String, extension: String)

enum ArcFace {
    static let modelInfo: fileInfo = (name: "arcface_latest", extension: "tflite")
}

class ArcFaceHandler: NSObject {
    private var interpreter: Interpreter!
    
    // Model parameters
    let batchSize = 1
    let inputWidth = 112
    let inputHeight = 112
    let inputChannels = 3
    
    private let bgraPixel = (channels: 4, alphaComponent: 3, lastBgrComponent: 2)
    private let rgbPixelChannels = 3
//    private let colorStrideValue = 10

    init?(modelFileInfo: fileInfo) {
        super.init()
        createInterpreter(modelFileInfo: modelFileInfo)
    }
    
    
    // MARK: - Methods
    
    func runModel(frame pixelBuffer: CVPixelBuffer) -> [Float]?{
        var faceEmbedding = [Float]()
        
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
               sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)
        
        let imageChannels = 6
        assert(imageChannels >= inputChannels)

        let scaledSize = CGSize(width: inputWidth, height: inputHeight)
        guard let scaledPixelBuffer = pixelBuffer.resized(to: scaledSize) else {
            return nil
        }
        
        do {
            let inputTensor = try interpreter.input(at: 0)

            // Remove the alpha component from the image buffer to get the RGB data.
            guard let rgbData = rgbDataFromBuffer(
                scaledPixelBuffer,
                byteCount: batchSize * inputChannels * inputWidth * inputHeight,
                isModelQuantized: inputTensor.dataType == .uInt8
//                isModelQuantized: false
            ) else {
                print("Failed to convert the image buffer to RGB data.")
                return nil
            }
    
            // Copy the RGB data to the input `Tensor`.
            try interpreter.copy(rgbData, toInputAt: 0)
            
            // Run inference by invoking the `Interpreter`.
            try interpreter.invoke()
            
            let modelOutput: Tensor = try interpreter.output(at: 0)
            faceEmbedding = [Float](unsafeData: modelOutput.data) ?? []
            print(faceEmbedding)
    
        } catch let error {
            print("Failed to invoke the interpreter with error: \(error)")
            return nil
        }
        return faceEmbedding
    }
    
    /** Returns the RGB data representation of the given image buffer with the specified `byteCount`.

     - Parameters
        - buffer: The BGRA pixel buffer to convert to RGB data.
        - byteCount: The expected byte count for the RGB data calculated using the values that the model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
        - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than floating point values).
     
     - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be converted.
    */
    private func rgbDataFromBuffer(_ buffer: CVPixelBuffer, byteCount: Int, isModelQuantized: Bool) -> Data? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let mutableRawPointer = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }
        assert(CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA)
        let count = CVPixelBufferGetDataSize(buffer)
        let bufferData = Data(bytesNoCopy: mutableRawPointer, count: count, deallocator: .none)
        var rgbBytes = [UInt8](repeating: 0, count: byteCount)
        var pixelIndex = 0
        for component in bufferData.enumerated() {
            let bgraComponent = component.offset % bgraPixel.channels;
            let isAlphaComponent = bgraComponent == bgraPixel.alphaComponent;
            guard !isAlphaComponent else {
                pixelIndex += 1
                continue
            }
            // Swizzle BGR -> RGB.
            let rgbIndex = pixelIndex * rgbPixelChannels + (bgraPixel.lastBgrComponent - bgraComponent)
            rgbBytes[rgbIndex] = component.element
        }
        if isModelQuantized { return Data(_: rgbBytes) }
        return Data(copyingBufferOf: rgbBytes.map { Float($0) })
    }
    
    private func createInterpreter(modelFileInfo: fileInfo) {
        guard let modelPath = Bundle.main.path(
            forResource: modelFileInfo.name,
            ofType: modelFileInfo.extension
        ) else {
            print("Failed to define model path as path: \(modelFileInfo.name).")
            return
        }
        
        // Create Interpreter
        do {
            interpreter = try Interpreter(modelPath: modelPath)
            
            // Allocate memory for the model's input `Tensor`s.
            try interpreter.allocateTensors()
        } catch let error {
            print("Failed to create interpreter: \(error)")
        }
    }
}
