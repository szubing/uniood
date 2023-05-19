from templates.template_pool import ALL_TEMPLATES
from templates.template_mining import MINED_TEMPLATES
from templates.hand_crafted import TIP_ADAPTER_TEMPLATES

templates_types = ['classname', 'vanilla', 'hand_crafted', 'ensemble', 'template_mining']

def get_templates(dataset_name, text_augmentation):
    """Return a list of templates to use for the given config."""
    if text_augmentation == 'classname':
        return ["{}"]
    elif text_augmentation == 'vanilla':
        return ["a photo of a {}."]
    elif text_augmentation == 'hand_crafted':
        return TIP_ADAPTER_TEMPLATES[dataset_name]
    elif text_augmentation == 'ensemble':
        return ALL_TEMPLATES
    elif text_augmentation == 'template_mining':
        return MINED_TEMPLATES[dataset_name]
    else:
        raise ValueError('Unknown template: {}'.format(text_augmentation))