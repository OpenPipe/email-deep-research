# %%
import dotenv

dotenv.load_dotenv()

# %%
# !uv run aws s3 sync s3://email-deep-research-backups/email_agent/models/email-agent-008/0594/ /tmp/agent-008


# %%

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")


peft_model = PeftModel.from_pretrained(base, "/tmp/agent-008")

# %%
peft_model

# %%

peft_model.push_to_hub("OpenPipe/art-e-008", private=False)
tokenizer.push_to_hub("OpenPipe/art-e-008", private=False)
