	?Ea???@?Ea???@!?Ea???@	Nj?????Nj?????!Nj?????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?Ea???@F#?W<???A:?61??@Ys.?Ue?@*	?????Hx@2F
Iterator::Model???R$_??!???W@)???????1?9?d?R@:Preprocessing2U
Iterator::Model::ForeverRepeat?????Q??!?(c?j2@)M?O????1?p/ ?-@:Preprocessing2n
7Iterator::Model::ForeverRepeat::Prefetch::ParallelMapV2pz?????!^e??~@)pz?????1^e??~@:Preprocessing2_
(Iterator::Model::ForeverRepeat::Prefetch%#gaO;??!??X?a@)%#gaO;??1??X?a@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Oj?????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	F#?W<???F#?W<???!F#?W<???      ??!       "      ??!       *      ??!       2	:?61??@:?61??@!:?61??@:      ??!       B      ??!       J	s.?Ue?@s.?Ue?@!s.?Ue?@R      ??!       Z	s.?Ue?@s.?Ue?@!s.?Ue?@JCPU_ONLYYOj?????b 