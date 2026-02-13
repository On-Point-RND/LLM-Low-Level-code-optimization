from prompt_generators.prompt_registry import register_prompt, BasePromptStrategy
from utils.utils import underscore_to_pascalcase
from prompt_generators.prompt_utils import read_relavant_files, ascendc_template



@register_prompt("ascendc", "add_shot")
class AscendcDefaultPromptStrategy(BasePromptStrategy):        
    def generate(self, op):
        arc_src, example_arch_src, example_new_arch_src = read_relavant_files('ascendc', op, 'add')
        return ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, 'add')

