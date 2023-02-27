import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text


class GlassDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.glass_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']


        #/data/wangdonghui/datasets/glass/1_XML/0_CenterView.png
        # /data/wangdonghui/datasets/glass/1_XML/0_CenterView.png
        frames = ['{base_path}/{sequence_path}/img/{frame}{nz}.{ext}'.format(base_path=self.base_path,
        sequence_path=sequence_path,frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame , end_frame + 1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        # NOTE: OTB has some weird annos which panda cannot handle
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'glass', ground_truth_rect[0:, :],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "1_XML", "path": "1_XML", "startFrame": 0, "endFrame": 99, "nz": "_CenterView", "ext": "png",
             "anno_path": "1_XML/groundtruth_rect.txt", "object_class": "glass1"},
            # {"name": "2_XML", "path": "2_XML", "startFrame": 0, "endFrame": 99, "nz": "_CenterView", "ext": "png",
            #  "anno_path": "2_XML/groundtruth_rect.txt", "object_class": "glass2"},
            {"name": "3_XML", "path": "3_XML", "startFrame": 0, "endFrame": 99, "nz": "_CenterView", "ext": "png",
             "anno_path": "3_XML/groundtruth_rect.txt", "object_class": "glass3"},
            {"name": "4_XML", "path": "4_XML", "startFrame": 0, "endFrame": 99, "nz": "_CenterView", "ext": "png",
             "anno_path": "4_XML/groundtruth_rect.txt", "object_class": "glass4"},
            {"name": "5_XML", "path": "5_XML", "startFrame": 0, "endFrame": 99, "nz": "_CenterView", "ext": "png",
             "anno_path": "5_XML/groundtruth_rect.txt", "object_class": "glass5"},
            {"name": "6_XML", "path": "6_XML", "startFrame": 0, "endFrame": 99, "nz": "_CenterView", "ext": "png",
             "anno_path": "6_XML/groundtruth_rect.txt", "object_class": "glass6"},
            # {"name": "7_XML", "path": "7_XML", "startFrame": 0, "endFrame": 99, "nz": "_CenterView", "ext": "png",
            #  "anno_path": "7_XML/groundtruth_rect.txt", "object_class": "glass7"},
            {"name": "8_XML", "path": "8_XML", "startFrame": 0, "endFrame": 99, "nz": "_CenterView", "ext": "png",
             "anno_path": "8_XML/groundtruth_rect.txt", "object_class": "glass8"},
            {"name": "9_XML", "path": "9_XML", "startFrame": 0, "endFrame": 99, "nz": "_CenterView", "ext": "png",
             "anno_path": "9_XML/groundtruth_rect.txt", "object_class": "glass9"},
            {"name": "10_XML", "path": "10_XML", "startFrame": 0, "endFrame": 69, "nz": "_CenterView", "ext": "png",
             "anno_path": "10_XML/groundtruth_rect.txt", "object_class": "glass10"}
        ]

        return sequence_info_list

if __name__ == '__main__':
    dataset = GlassDataset()
    sequence_list = dataset.get_sequence_list()
    print("sequence_list=",sequence_list)