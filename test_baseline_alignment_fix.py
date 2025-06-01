#!/usr/bin/env python3
"""
Test script to validate the baseline cultural alignment scoring fix.
"""

import sys
import time
from utility.baseline import generate_baseline_essay
from utility.cultural_alignment import derive_relevant_cultures
from run_100_cycles_final import evaluate_baseline_response

def test_baseline_alignment_fix():
    """Test the baseline cultural alignment fix with sample responses."""
    print("Testing Baseline Cultural Alignment Fix")
    print("=" * 50)
    
    # Test user profiles
    test_profiles = [
        {
            "place of birth": "California/CA",
            "ethnicity": "Mexican",
            "household language": "Spanish"
        },
        {
            "place of birth": "Texas/TX", 
            "ethnicity": "American",
            "household language": "English"
        },
        {
            "place of birth": "New York/NY",
            "ethnicity": "Chinese",
            "household language": "Mandarin"
        }
    ]
    
    # Test questions
    test_questions = [
        "Should children always respect their elders?",
        "Is it important for families to live close together?",
        "How important is individual achievement vs community harmony?"
    ]
    
    print("Testing baseline evaluation with different response types...\n")
    
    # Test 1: Sample baseline responses (culturally neutral)
    sample_responses = [
        # Response 1: High cultural concept awareness
        """Respect for elders is a complex topic that varies across different communities and families. 
        Many people consider the wisdom and experience of older generations valuable, while others 
        believe respect should be earned through actions rather than age alone. A balanced approach 
        might involve honoring the perspectives of elders while maintaining individual autonomy 
        and critical thinking. Cultural traditions often emphasize reverence for authority figures, 
        but modern society also values personal independence and diverse viewpoints.""",
        
        # Response 2: Medium cultural concept awareness  
        """Family proximity is an important consideration for many people. Some families benefit 
        from living close together, sharing resources and support systems. Others find that 
        distance allows for personal growth and independence. The best approach depends on 
        individual circumstances, family dynamics, and personal preferences. Different communities 
        have various traditions regarding family structure and living arrangements.""",
        
        # Response 3: Low cultural concept awareness
        """Individual achievement and community harmony both have merit. People should strive 
        to do their best while also being considerate of others. Success can be measured in 
        different ways, and what works for one person may not work for another. Balance is key."""
    ]
    
    # Test each combination
    for i, (profile, question, response) in enumerate(zip(test_profiles, test_questions, sample_responses)):
        print(f"Test {i+1}:")
        print(f"User: {profile['ethnicity']} from {profile['place of birth']}")
        print(f"Question: {question}")
        print(f"Response preview: {response[:100]}...")
        
        # Get user's relevant cultures
        relevant_cultures = derive_relevant_cultures(profile)
        print(f"User's relevant cultures: {relevant_cultures}")
        
        # Evaluate the response
        metrics = evaluate_baseline_response(response, profile)
        cultural_score = metrics['cultural_alignment_score']
        unique_cultures = metrics['unique_cultures']
        
        print(f"Cultural alignment score: {cultural_score:.3f}")
        print(f"Unique cultural terms: {unique_cultures}")
        print(f"Response length: {metrics['avg_response_length']}")
        print("-" * 40)
    
    # Test 4: Generate actual baseline response and evaluate
    print("\nTesting with actual baseline generation:")
    test_profile = test_profiles[0]
    test_question = test_questions[0]
    
    print(f"Generating baseline for: {test_question}")
    print(f"User: {test_profile}")
    
    try:
        # Generate baseline response
        baseline_response = generate_baseline_essay([test_profile], test_question)
        print(f"Generated response: {baseline_response[:200]}...")
        
        # Evaluate it
        metrics = evaluate_baseline_response(baseline_response, test_profile)
        
        print(f"\nBaseline Evaluation Results:")
        print(f"  Cultural alignment score: {metrics['cultural_alignment_score']:.3f}")
        print(f"  Response length: {metrics['avg_response_length']}")
        print(f"  Unique cultural terms: {metrics['unique_cultures']}")
        print(f"  Response completeness: {metrics['response_completeness']:.3f}")
        
        # Check if score is meaningful (not 0.00)
        if metrics['cultural_alignment_score'] > 0:
            print("✅ SUCCESS: Baseline now produces meaningful cultural alignment scores!")
        else:
            print("❌ ISSUE: Baseline still producing 0.00 cultural alignment scores")
            
    except Exception as e:
        print(f"❌ ERROR generating baseline: {e}")
    
    print("\n" + "=" * 50)
    print("Baseline Cultural Alignment Fix Test Complete")

if __name__ == "__main__":
    test_baseline_alignment_fix()
