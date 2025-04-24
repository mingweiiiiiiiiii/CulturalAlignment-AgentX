import pytest
import os
import numpy as np
from ..inputData import PersonaSampler
from inputData import (
    process_user_input,
    validate_user_profile,
    process_wvs_data,
    normalize_demographics
)

@pytest.fixture
def sampler():
    return PersonaSampler()

def test_persona_sampler_initialization(sampler):
    assert sampler is not None
    assert sampler.question_to_options is not None
    assert sampler.sampling_fields is not None

def test_sample_profiles(sampler):
    # Test single profile generation
    profiles = sampler.sample_profiles(n=1)
    assert len(profiles) == 1
    profile = profiles[0]
    
    # Verify all required fields are present
    required_fields = [
        "sex", "age", "marital_status", "education",
        "employment_sector", "social_class", "income_level",
        "ethnicity", "country"
    ]
    for field in required_fields:
        assert field in profile
    
    # Test multiple profile generation
    profiles = sampler.sample_profiles(n=3)
    assert len(profiles) == 3

def test_sample_question(sampler):
    question, options = sampler.sample_question()
    assert isinstance(question, str)
    assert isinstance(options, list)
    assert len(options) > 0

def test_build_prompt(sampler):
    # Create a test profile
    profile = {
        "sex": "Male",
        "age": 30,
        "marital_status": "Single",
        "education": "Bachelor's degree",
        "employment_sector": "Technology",
        "social_class": "Middle class",
        "income_level": "Medium",
        "ethnicity": "Asian",
        "country": "Japan"
    }
    
    question = "What is your view on traditional customs?"
    options = ["Very important", "Somewhat important", "Not important"]
    
    prompt = sampler.build_prompt(profile, question, options)
    
    # Verify prompt contains all necessary information
    assert isinstance(prompt, str)
    assert profile["sex"] in prompt
    assert str(profile["age"]) in prompt
    assert profile["country"] in prompt
    assert question in prompt
    for option in options:
        assert option in prompt

def test_error_handling():
    # Test with invalid paths
    with pytest.raises(Exception):
        PersonaSampler(wvs_path='nonexistent.json', persona_path='nonexistent.yml')
    
    # Test with malformed WVS data
    with pytest.raises(KeyError):
        bad_sampler = PersonaSampler(wvs_path='bad_wvs.json')

def test_sampling_fields_content(sampler):
    fields = sampler.get_sampling_fields()
    
    # Verify required fields exist
    required_fields = [
        "Sex", "Age", "Marital Status", "Education",
        "Employment Sector", "Social Class", "Income Level",
        "Ethnicity", "Country"
    ]
    for field in required_fields:
        assert field in fields
    
    # Verify age range
    assert isinstance(fields["Age"], list)
    assert min(fields["Age"]) >= 18
    assert max(fields["Age"]) <= 70

def test_profile_values_validity(sampler):
    profile = sampler.sample_profiles(n=1)[0]
    fields = sampler.get_sampling_fields()
    
    # Verify each field's value is from the allowed set
    assert profile["sex"] in fields["Sex"]
    assert profile["age"] in fields["Age"]
    assert profile["marital_status"] in fields["Marital Status"]
    assert profile["education"] in fields["Education"]
    assert profile["employment_sector"] in fields["Employment Sector"]
    assert profile["social_class"] in fields["Social Class"]
    assert profile["income_level"] in fields["Income Level"]
    assert profile["ethnicity"] in fields["Ethnicity"]
    assert profile["country"] in fields["Country"]

def test_file_loading_errors():
    # Test nonexistent WVS file
    with pytest.raises(RuntimeError, match="Failed to load WVS questions"):
        PersonaSampler(wvs_path='nonexistent.json')
    
    # Test nonexistent persona file
    with pytest.raises(RuntimeError, match="Failed to load persona template"):
        PersonaSampler(persona_path='nonexistent.yml')
    
    # Test malformed WVS JSON
    with open('test_malformed.json', 'w') as f:
        f.write('{invalid json')
    try:
        with pytest.raises(RuntimeError):
            PersonaSampler(wvs_path='test_malformed.json')
    finally:
        os.remove('test_malformed.json')
    
    # Test malformed YAML
    with open('test_malformed.yml', 'w') as f:
        f.write('invalid: yaml:\nstructure')
    try:
        with pytest.raises(RuntimeError):
            PersonaSampler(persona_path='test_malformed.yml')
    finally:
        os.remove('test_malformed.yml')

def test_invalid_persona_fields(sampler):
    # Test invalid age
    profile = {
        "sex": "Male",
        "age": "invalid",  # Should be int
        "marital_status": "Single",
        "education": "Bachelor's degree",
        "employment_sector": "Technology",
        "social_class": "Middle class",
        "income_level": "Medium",
        "ethnicity": "Asian",
        "country": "Japan"
    }
    with pytest.raises(ValueError, match="Age must be a number"):
        sampler.build_prompt(profile, "test question", ["option1"])
    
    # Test invalid country
    profile["age"] = 30
    profile["country"] = "NonexistentCountry"
    with pytest.raises(ValueError, match="Invalid country"):
        sampler.build_prompt(profile, "test question", ["option1"])

def test_prompt_building_variations(sampler):
    base_profile = {
        "sex": "Male",
        "age": 30,
        "marital_status": "Single",
        "education": "Bachelor's degree",
        "employment_sector": "Technology",
        "social_class": "Middle class",
        "income_level": "Medium",
        "ethnicity": "Asian",
        "country": "Japan"
    }
    
    # Test with minimal options
    prompt = sampler.build_prompt(base_profile, "Test question?", ["Yes", "No"])
    assert isinstance(prompt, str)
    assert "Test question?" in prompt
    assert "Yes" in prompt
    assert "No" in prompt
    
    # Test with many options
    many_options = [f"Option {i}" for i in range(10)]
    prompt = sampler.build_prompt(base_profile, "Test question?", many_options)
    assert isinstance(prompt, str)
    for opt in many_options:
        assert opt in prompt
    
    # Test with special characters
    special_question = "What's your opinion on this? (考えは何ですか?)"
    prompt = sampler.build_prompt(base_profile, special_question, ["はい", "いいえ"])
    assert isinstance(prompt, str)
    assert special_question in prompt
    assert "はい" in prompt
    assert "いいえ" in prompt

def test_profile_sampling_distribution(sampler):
    # Test distribution of sampled fields
    n_samples = 1000
    profiles = sampler.sample_profiles(n=n_samples)
    
    # Check age distribution
    ages = [p["age"] for p in profiles]
    assert min(ages) >= 18
    assert max(ages) <= 70
    
    # Check country distribution
    countries = [p["country"] for p in profiles]
    unique_countries = set(countries)
    # Should have reasonable diversity in a large sample
    assert len(unique_countries) > 10
    
    # Check sex distribution
    sexes = [p["sex"] for p in profiles]
    male_ratio = sum(1 for s in sexes if s == "Male") / n_samples
    # Should be roughly balanced
    assert 0.4 <= male_ratio <= 0.6

def test_question_sampling_uniqueness(sampler):
    # Test that we get different questions
    questions = set()
    for _ in range(50):
        q, _ = sampler.sample_question()
        questions.add(q)
    # Should get reasonable variety in 50 samples
    assert len(questions) > 10

def test_sampling_fields_validation(sampler):
    fields = sampler.get_sampling_fields()
    
    # Check required fields
    required = ["Sex", "Age", "Marital Status", "Education", 
               "Employment Sector", "Social Class", "Income Level",
               "Ethnicity", "Country"]
    for field in required:
        assert field in fields
    
    # Check field content
    assert isinstance(fields["Age"], list)
    assert all(isinstance(x, int) for x in fields["Age"])
    assert isinstance(fields["Sex"], list)
    assert all(isinstance(x, str) for x in fields["Sex"])
    
    # Check country list
    assert isinstance(fields["Country"], list)
    assert len(fields["Country"]) > 0
    assert all(isinstance(x, str) for x in fields["Country"])

def test_process_user_input(mock_user_profile):
    input_text = "How do different cultures celebrate holidays?"
    result = process_user_input(input_text, mock_user_profile)
    
    assert "question_meta" in result
    assert "original" in result["question_meta"]
    assert result["question_meta"]["original"] == input_text
    assert "user_profile" in result
    assert result["user_profile"] == mock_user_profile

def test_validate_user_profile():
    # Test valid profile
    valid_profile = {
        "demographics": {
            "country": "US",
            "age": 30,
            "sex": "Female",
            "education": "Bachelor's degree"
        },
        "preferences": {}
    }
    assert validate_user_profile(valid_profile) == True
    
    # Test invalid profiles
    invalid_profiles = [
        {},  # Empty profile
        {"demographics": {}},  # Empty demographics
        {  # Missing required field
            "demographics": {
                "country": "US",
                "sex": "Female"
            },
            "preferences": {}
        },
        {  # Invalid age
            "demographics": {
                "country": "US",
                "age": "thirty",
                "sex": "Female",
                "education": "Bachelor's degree"
            },
            "preferences": {}
        }
    ]
    
    for profile in invalid_profiles:
        with pytest.raises(Exception):
            validate_user_profile(profile)

def test_process_wvs_data():
    # Mock WVS data
    mock_wvs_data = {
        "US": {
            "values": np.random.rand(10),
            "demographics": {
                "age_distribution": [0.2, 0.3, 0.3, 0.2],
                "education_levels": {"high": 0.4, "medium": 0.4, "low": 0.2}
            }
        }
    }
    
    result = process_wvs_data(mock_wvs_data)
    assert isinstance(result, dict)
    assert "US" in result
    assert "values" in result["US"]
    assert "demographics" in result["US"]

def test_normalize_demographics():
    # Test age normalization
    age = 30
    normalized_age = normalize_demographics("age", age)
    assert isinstance(normalized_age, float)
    assert 0 <= normalized_age <= 1
    
    # Test education normalization
    education_levels = [
        "No formal education",
        "Primary school",
        "Secondary school",
        "Bachelor's degree",
        "Master's degree",
        "Doctorate"
    ]
    
    for edu in education_levels:
        normalized_edu = normalize_demographics("education", edu)
        assert isinstance(normalized_edu, float)
        assert 0 <= normalized_edu <= 1
    
    # Test invalid inputs
    with pytest.raises(ValueError):
        normalize_demographics("age", -1)
    with pytest.raises(ValueError):
        normalize_demographics("education", "Invalid degree")
    with pytest.raises(ValueError):
        normalize_demographics("invalid_field", "some value")