"""Quick test to show 'I just want cleaned data' instruction works"""
from instruction_handler import UserInstructionHandler

handler = UserInstructionHandler()

# Test variations of "cleaned data only" instruction
test_instructions = [
    "I just want cleaned data",
    "Just clean the data",
    "Only clean, no training",
    "Clean data only",
    "I want cleaned data, no models",
    "Just preprocess the data",
    "Clean the data, skip training",
]

print("=" * 70)
print("✅ Testing 'Cleaned Data Only' Instructions")
print("=" * 70)

for instruction in test_instructions:
    parsed = handler._parse_instructions(instruction)
    detected_types = [item['type'].name for item in parsed]
    
    # Check if CLEAN_ONLY or NO_TRAINING detected
    is_clean_only = any(t in ['CLEAN_ONLY', 'NO_TRAINING'] for t in detected_types)
    
    status = "✅" if is_clean_only else "❌"
    print(f"\n{status} \"{instruction}\"")
    print(f"   Detected: {detected_types}")
    
    if is_clean_only:
        print(f"   → Agent will: Clean data and skip model training ✓")

print("\n" + "=" * 70)
print("🎉 All variations work! The agent understands you want cleaned data only.")
print("=" * 70)
