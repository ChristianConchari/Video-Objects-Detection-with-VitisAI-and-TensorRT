	?hUK:?^@?hUK:?^@!?hUK:?^@	?ʹC?????ʹC????!?ʹC????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?hUK:?^@vQ?????A?????^@Y\ ?K???*	-?????L@2U
Iterator::Model::ForeverRepeatd??????!???\/YS@)??Q????1D??e??M@:Preprocessing2_
(Iterator::Model::ForeverRepeat::Prefetch}zlˀ???!?&䧾|1@)}zlˀ???1?&䧾|1@:Preprocessing2n
7Iterator::Model::ForeverRepeat::Prefetch::ParallelMapV2?2d????!ʳkmӇ0@)?2d????1ʳkmӇ0@:Preprocessing2F
Iterator::Model
?B?Գ??!?$?T@)]???2?l?1T|?M@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?ʹC????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	vQ?????vQ?????!vQ?????      ??!       "      ??!       *      ??!       2	?????^@?????^@!?????^@:      ??!       B      ??!       J	\ ?K???\ ?K???!\ ?K???R      ??!       Z	\ ?K???\ ?K???!\ ?K???JCPU_ONLYY?ʹC????b 