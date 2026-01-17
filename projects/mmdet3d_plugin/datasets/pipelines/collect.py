from mmengine.registry import TRANSFORMS

from projects.mmdet3d_plugin.compat import DataContainer as DC


@TRANSFORMS.register_module()
class Collect(object):
    """Collect data from the loader pipeline and format into a new dict."""

    def __init__(self, keys, meta_keys=None, meta_name="img_metas"):
        self.keys = keys
        self.meta_keys = meta_keys or []
        self.meta_name = meta_name

    def __call__(self, results):
        data = {}
        img_metas = {key: results[key] for key in self.meta_keys}
        data[self.meta_name] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(keys={self.keys}, "
            f"meta_keys={self.meta_keys}, meta_name={self.meta_name})"
        )
