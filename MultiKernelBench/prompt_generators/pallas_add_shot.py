from prompt_generators.prompt_registry import register_prompt, BasePromptStrategy
from prompt_generators.prompt_utils import generate_template, read_relavant_files

@register_prompt("pallas", "add_shot")
class PallasDefaultPromptStrategy(BasePromptStrategy):
    def generate(self, op) -> str:
        arc_src, example_arch_src, example_new_arch_src = read_relavant_files('pallas', op, 'add')
        return generate_template(arc_src, example_arch_src, example_new_arch_src, 'pallas')