import UIKit
import Metal


@available(iOS 13.0, *)
class ViewController: UIViewController {
  
  
  override func viewDidLoad() {
    super.viewDidLoad()
      
    let count = 10000000
    let elementsPerSum = 10000

    // Data type, has to be the same as in the shader
    typealias DataType = CInt

    let device = MTLCreateSystemDefaultDevice()!
    let library = device.makeDefaultLibrary()

    let parsum = library!.makeFunction(name: "parsum")!
    let pipeline = try! device.makeComputePipelineState(function: parsum)

    // Our data, randomly generated:
    var data = (0..<count).map{ _ in DataType(arc4random_uniform(100)) }

    var dataCount = CUnsignedInt(count)
    var elementsPerSumC = CUnsignedInt(elementsPerSum)
    // Number of individual results = count / elementsPerSum (rounded up):
    let resultsCount = (count + elementsPerSum - 1) / elementsPerSum

    // Our data in a buffer (copied):
    let dataBuffer = device.makeBuffer(bytes: &data, length: MemoryLayout<DataType>.stride * count, options: [])!
    // A buffer for individual results (zero initialized)
    let resultsBuffer = device.makeBuffer(length: MemoryLayout<DataType>.stride * resultsCount, options: [])!
    // Our results in convenient form to compute the actual result later:
    let pointer = resultsBuffer.contents().bindMemory(to: DataType.self, capacity: resultsCount)
    let results = UnsafeBufferPointer<DataType>(start: pointer, count: resultsCount)

    let queue = device.makeCommandQueue()!
    let cmds = queue.makeCommandBuffer()!
    let encoder = cmds.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)

    encoder.setBuffer(dataBuffer, offset: 0, index: 0)

    encoder.setBytes(&dataCount, length: MemoryLayout<CUnsignedInt>.size, index: 1)
    encoder.setBuffer(resultsBuffer, offset: 0, index: 2)
    encoder.setBytes(&elementsPerSumC, length: MemoryLayout<CUnsignedInt>.size, index: 3)

    // We have to calculate the sum `resultCount` times => amount of threadgroups is `resultsCount` / `threadExecutionWidth` (rounded up) because each threadgroup will process `threadExecutionWidth` threads
    let threadgroupsPerGrid = MTLSize(width: (resultsCount + pipeline.threadExecutionWidth - 1) / pipeline.threadExecutionWidth, height: 1, depth: 1)

    // Here we set that each threadgroup should process `threadExecutionWidth` threads, the only important thing for performance is that this number is a multiple of `threadExecutionWidth` (here 1 times)
    let threadsPerThreadgroup = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)

    encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()

    var start, end : UInt64
    var result : DataType = 0

    start = mach_absolute_time()
    cmds.commit()
    cmds.waitUntilCompleted()
    for elem in results {
        result += elem
    }

    end = mach_absolute_time()

    print("Metal result: \(result), time: \(Double(end - start) / Double(NSEC_PER_SEC))")
    result = 0

    start = mach_absolute_time()
    data.withUnsafeBufferPointer { buffer in
        for elem in buffer {
            result += elem
        }
    }
    end = mach_absolute_time()

    print("CPU result: \(result), time: \(Double(end - start) / Double(NSEC_PER_SEC))")
    
  }


}
