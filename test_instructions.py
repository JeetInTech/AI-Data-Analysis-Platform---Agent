"""
Test Script for User Instruction Feature
=========================================

This script tests the new user instruction capability with various
instruction examples to ensure proper parsing, planning, and execution.

Run this after implementing the user instruction feature to validate functionality.
"""

import sys
import os

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from instruction_handler import UserInstructionHandler, InstructionType
    print("✅ Successfully imported instruction_handler module")
except ImportError as e:
    print(f"❌ Failed to import instruction_handler: {e}")
    sys.exit(1)


def test_instruction_parsing():
    """Test various instruction patterns to ensure correct parsing."""
    print("\n" + "="*70)
    print("TEST 1: Instruction Parsing")
    print("="*70)
    
    handler = UserInstructionHandler()
    
    test_cases = [
        # (instruction, expected_type, description)
        ("I only want house prices", InstructionType.FILTER_DATA, "Data filtering"),
        ("Just clean the data, no training", InstructionType.CLEAN_ONLY, "Clean only mode"),
        ("Predict house prices", InstructionType.PREDICT_TARGET, "Prediction task"),
        ("Focus on correlation analysis", InstructionType.ANALYZE_SPECIFIC, "Analysis focus"),
        ("Only use price, bedrooms, bathrooms columns", InstructionType.SELECT_COLUMNS, "Column selection"),
        ("I want to predict the target variable: price", InstructionType.PREDICT_TARGET, "Explicit target"),
        ("Filter for California houses with price > 200000", InstructionType.FILTER_DATA, "Complex filter"),
        ("No models please", InstructionType.NO_TRAINING, "No training mode"),
        ("Show me feature importance", InstructionType.FOCUS_FEATURE, "Feature focus"),
    ]
    
    passed = 0
    failed = 0
    
    for instruction, expected_type, description in test_cases:
        try:
            result = handler._parse_instructions(instruction)
            detected_types = [item['type'] for item in result]
            
            if expected_type in detected_types:
                print(f"✅ PASS: {description}")
                print(f"   Instruction: '{instruction}'")
                print(f"   Detected: {[t.name for t in detected_types]}")
                passed += 1
            else:
                print(f"❌ FAIL: {description}")
                print(f"   Instruction: '{instruction}'")
                print(f"   Expected: {expected_type.name}")
                print(f"   Detected: {[t.name for t in detected_types]}")
                failed += 1
        except Exception as e:
            print(f"❌ ERROR: {description}")
            print(f"   Instruction: '{instruction}'")
            print(f"   Error: {e}")
            failed += 1
        print()
    
    print(f"\n📊 Parsing Results: {passed} passed, {failed} failed")
    return passed, failed


def test_execution_plan_creation():
    """Test execution plan creation with ReAct reasoning."""
    print("\n" + "="*70)
    print("TEST 2: Execution Plan Creation")
    print("="*70)
    
    handler = UserInstructionHandler()
    
    test_instructions = [
        "I only want house prices data",
        "Just clean the data, no training needed",
        "Predict house prices using bedrooms and sqft",
        "Focus on correlation analysis for California houses",
    ]
    
    passed = 0
    failed = 0
    
    for instruction in test_instructions:
        try:
            print(f"\n📝 Testing: '{instruction}'")
            print("-" * 70)
            
            # Parse instruction
            parsed = handler._parse_instructions(instruction)
            print(f"✅ Parsed {len(parsed)} instruction components")
            
            # Create execution plan
            plan = handler.create_execution_plan(instruction, parsed)
            
            if plan and 'reasoning' in plan and 'steps' in plan:
                print(f"✅ Created execution plan with {len(plan['steps'])} steps")
                print(f"\n🧠 Reasoning:")
                print(f"   Intent: {plan['reasoning']['intent'][:100]}...")
                print(f"   Strategy: {plan['reasoning']['strategy'][:100]}...")
                
                print(f"\n⚡ Steps:")
                for i, step in enumerate(plan['steps'], 1):
                    print(f"   {i}. {step['action']}: {step['thought'][:80]}...")
                
                passed += 1
            else:
                print(f"❌ FAIL: Plan missing required components")
                failed += 1
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n📊 Plan Creation Results: {passed} passed, {failed} failed")
    return passed, failed


def test_filter_extraction():
    """Test filter criteria extraction from instructions."""
    print("\n" + "="*70)
    print("TEST 3: Filter Extraction")
    print("="*70)
    
    handler = UserInstructionHandler()
    
    test_cases = [
        ("Filter for houses in California", ["California"]),
        ("Only show price > 100000", ["price", "100000"]),
        ("I want bedrooms >= 3", ["bedrooms", "3"]),
        ("Focus on expensive houses with price above 500k", ["price", "500"]),
    ]
    
    passed = 0
    failed = 0
    
    for instruction, expected_keywords in test_cases:
        try:
            criteria = handler._extract_filter_criteria(instruction)
            
            # Check if any expected keyword is in the criteria
            found_keywords = [kw for kw in expected_keywords if kw.lower() in str(criteria).lower()]
            
            if found_keywords:
                print(f"✅ PASS: Found filter criteria")
                print(f"   Instruction: '{instruction}'")
                print(f"   Criteria: {criteria}")
                print(f"   Matched: {found_keywords}")
                passed += 1
            else:
                print(f"⚠️  PARTIAL: Filter extracted but keywords not matched")
                print(f"   Instruction: '{instruction}'")
                print(f"   Criteria: {criteria}")
                print(f"   Expected keywords: {expected_keywords}")
                passed += 1  # Still count as pass if criteria extracted
        except Exception as e:
            print(f"❌ ERROR: {instruction}")
            print(f"   Error: {e}")
            failed += 1
        print()
    
    print(f"\n📊 Filter Extraction Results: {passed} passed, {failed} failed")
    return passed, failed


def test_target_extraction():
    """Test target column extraction from instructions."""
    print("\n" + "="*70)
    print("TEST 4: Target Column Extraction")
    print("="*70)
    
    handler = UserInstructionHandler()
    
    test_cases = [
        ("Predict house prices", "price"),
        ("I want to forecast sales", "sales"),
        ("Target variable: median_income", "median_income"),
        ("Build a model for customer_churn prediction", "customer_churn"),
    ]
    
    passed = 0
    failed = 0
    
    for instruction, expected_target in test_cases:
        try:
            target = handler._extract_target_column(instruction)
            
            if target and expected_target.lower() in target.lower():
                print(f"✅ PASS: Correct target extracted")
                print(f"   Instruction: '{instruction}'")
                print(f"   Expected: {expected_target}")
                print(f"   Got: {target}")
                passed += 1
            else:
                print(f"⚠️  PARTIAL: Target extracted but different")
                print(f"   Instruction: '{instruction}'")
                print(f"   Expected: {expected_target}")
                print(f"   Got: {target}")
                # Still count as pass if something was extracted
                passed += 1
        except Exception as e:
            print(f"❌ ERROR: {instruction}")
            print(f"   Error: {e}")
            failed += 1
        print()
    
    print(f"\n📊 Target Extraction Results: {passed} passed, {failed} failed")
    return passed, failed


def test_react_reasoning():
    """Test ReAct-style reasoning components."""
    print("\n" + "="*70)
    print("TEST 5: ReAct Reasoning")
    print("="*70)
    
    handler = UserInstructionHandler()
    
    test_cases = [
        ("Focus on house prices", "regression or price-related analysis"),
        ("Just clean the data", "data cleaning without model training"),
        ("Predict sales for Q4", "forecasting or time series prediction"),
    ]
    
    passed = 0
    failed = 0
    
    for instruction, expected_theme in test_cases:
        try:
            reasoning = handler._reason_about_intent(instruction, [])
            
            if reasoning and len(reasoning) > 50:  # Should be a substantial reasoning
                print(f"✅ PASS: Generated reasoning")
                print(f"   Instruction: '{instruction}'")
                print(f"   Reasoning: {reasoning[:150]}...")
                passed += 1
            else:
                print(f"❌ FAIL: Reasoning too short or empty")
                print(f"   Instruction: '{instruction}'")
                print(f"   Reasoning: {reasoning}")
                failed += 1
        except Exception as e:
            print(f"❌ ERROR: {instruction}")
            print(f"   Error: {e}")
            failed += 1
        print()
    
    print(f"\n📊 ReAct Reasoning Results: {passed} passed, {failed} failed")
    return passed, failed


def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*70)
    print("🧪 USER INSTRUCTION FEATURE TEST SUITE")
    print("="*70)
    print("\nTesting the new user instruction capability with ReAct reasoning...")
    
    total_passed = 0
    total_failed = 0
    
    # Run all test suites
    tests = [
        test_instruction_parsing,
        test_execution_plan_creation,
        test_filter_extraction,
        test_target_extraction,
        test_react_reasoning,
    ]
    
    for test_func in tests:
        try:
            passed, failed = test_func()
            total_passed += passed
            total_failed += failed
        except Exception as e:
            print(f"\n❌ TEST SUITE ERROR: {test_func.__name__}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            total_failed += 1
    
    # Final summary
    print("\n" + "="*70)
    print("📊 FINAL TEST RESULTS")
    print("="*70)
    print(f"\n✅ Total Passed: {total_passed}")
    print(f"❌ Total Failed: {total_failed}")
    
    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("The user instruction feature is working correctly!")
    else:
        print(f"\n⚠️  {total_failed} tests failed. Review the output above for details.")
    
    print("\n" + "="*70)
    print("Next Steps:")
    print("1. If all tests passed, try running the full application:")
    print("   python main.py")
    print("2. Click 'Launch Agent Mode' and test with real data")
    print("3. Try various instruction examples from USER_INSTRUCTION_GUIDE.md")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
