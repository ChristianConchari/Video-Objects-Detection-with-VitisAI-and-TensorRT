?	c??ޖ|?@c??ޖ|?@!c??ޖ|?@	Ѕ?MW7??Ѕ?MW7??!Ѕ?MW7??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$c??ޖ|?@nk?KE??A?Բ]{?@Y??`?.???*??ʡI"?@?|?5^΂@)      ?=2?
xIterator::Model::ForeverRepeat::Prefetch::ParallelMapV2::ParallelMapV2::MapAndBatch::ParallelMapV2::FlatMap[0]::TFRecordePmp"?E@!I<?X@)ePmp"?E@1I<?X@:Advanced file read2U
Iterator::Model::ForeverRepeat??{G????!?mx????)d> Й???1?73?~??:Preprocessing2?
kIterator::Model::ForeverRepeat::Prefetch::ParallelMapV2::ParallelMapV2::MapAndBatch::ParallelMapV2::FlatMapa?????E@!è ???X@)k?SUh ??1/}??5??:Preprocessing2_
(Iterator::Model::ForeverRepeat::Prefetch!?????!?^??9???)!?????1?^??9???:Preprocessing2n
7Iterator::Model::ForeverRepeat::Prefetch::ParallelMapV2?|$%=??!2
?-Q??)?|$%=??12
?-Q??:Preprocessing2F
Iterator::Model~? ????!????5\??)???Y.??1? ??Δ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Ѕ?MW7??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	nk?KE??nk?KE??!nk?KE??      ??!       "      ??!       *      ??!       2	?Բ]{?@?Բ]{?@!?Բ]{?@:      ??!       B      ??!       J	??`?.?????`?.???!??`?.???R      ??!       Z	??`?.?????`?.???!??`?.???JCPU_ONLYYЅ?MW7??b Y      Y@q$ƽ????"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 