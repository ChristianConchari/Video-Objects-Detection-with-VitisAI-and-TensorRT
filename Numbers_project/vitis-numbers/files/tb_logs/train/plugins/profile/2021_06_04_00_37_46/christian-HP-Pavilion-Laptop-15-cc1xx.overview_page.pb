?	^??jG?^@^??jG?^@!^??jG?^@	????+t??????+t??!????+t??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$^??jG?^@?&??d??A0e????^@Y?͎T????*??????p@}?5^???@2?
bIterator::Model::ForeverRepeat::Prefetch::ParallelMapV2::ParallelMapV2::MapAndBatch::ParallelMapV2<H?}8@!???!?/U@)H?}8@1???!?/U@:Preprocessing2?
xIterator::Model::ForeverRepeat::Prefetch::ParallelMapV2::ParallelMapV2::MapAndBatch::ParallelMapV2::FlatMap[0]::TFRecord<5bf??(??!?Ϳ?&"@)5bf??(??1?Ϳ?&"@:Advanced file read2?
kIterator::Model::ForeverRepeat::Prefetch::ParallelMapV2::ParallelMapV2::MapAndBatch::ParallelMapV2::FlatMap<?'?bd???!??ޫ?(@)?gs???1veF???@:Preprocessing2U
Iterator::Model::ForeverRepeat?j,am???!?ǚCQ?@)k?MG ??1?r?T??:Preprocessing2_
(Iterator::Model::ForeverRepeat::Prefetch??x?&1??!T?G*n???)??x?&1??1T?G*n???:Preprocessing2n
7Iterator::Model::ForeverRepeat::Prefetch::ParallelMapV2?????P??!I??????)?????P??1I??????:Preprocessing2F
Iterator::Model???b?D??!????a@)?@H0?{?1#??????:Preprocessing2?
SIterator::Model::ForeverRepeat::Prefetch::ParallelMapV2::ParallelMapV2::MapAndBatch?6???Nt?!???{??)?6???Nt?1???{??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9????+t??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?&??d???&??d??!?&??d??      ??!       "      ??!       *      ??!       2	0e????^@0e????^@!0e????^@:      ??!       B      ??!       J	?͎T?????͎T????!?͎T????R      ??!       Z	?͎T?????͎T????!?͎T????JCPU_ONLYY????+t??b Y      Y@q????	,??"?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 