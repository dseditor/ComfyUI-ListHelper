"""
Show Text node for ComfyUI-ListHelper
Displays list-type text with proper segmentation
"""


class ShowTextListHelper:
    """
    A node that displays text with proper list formatting.
    Each list item is shown as a separate text widget.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "show_text"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "ListHelper/Utils"

    def show_text(self, text, unique_id=None, extra_pnginfo=None):
        """
        Display text and pass it through.
        Handles list input and stores widget values for workflow persistence.
        """
        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("Error: extra_pnginfo is not a list")
            elif (
                not isinstance(extra_pnginfo[0], dict)
                or "workflow" not in extra_pnginfo[0]
            ):
                print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),
                    None,
                )
                if node:
                    node["widgets_values"] = [text]

        return {"ui": {"text": text}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "ShowText|ListHelper": ShowTextListHelper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShowText|ListHelper": "Show Text (ListHelper)",
}
