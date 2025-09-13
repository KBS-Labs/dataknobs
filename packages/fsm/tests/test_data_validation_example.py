"""Tests for the data validation pipeline example.

This test validates that the data validation example works correctly,
including custom function registration, multi-stage validation, and
proper routing based on validation results.
"""

import pytest
import sys
from pathlib import Path

# Add examples directory to path
examples_dir = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(examples_dir))

from data_validation_pipeline import (
    normalize_input,
    validate_email,
    validate_phone,
    validate_age,
    calculate_validation_summary,
    config
)
from dataknobs_fsm.api.simple import SimpleFSM


class TestDataValidationPipeline:
    """Test the data validation pipeline example."""
    
    @pytest.fixture
    def fsm(self):
        """Create FSM with custom validation functions."""
        return SimpleFSM(
            config,
            custom_functions={
                'normalize_input': normalize_input,
                'validate_email': validate_email,
                'validate_phone': validate_phone,
                'validate_age': validate_age,
                'calculate_validation_summary': calculate_validation_summary
            }
        )
    
    def test_all_valid_data(self, fsm):
        """Test with all valid data."""
        result = fsm.process({
            "email": "john.doe@gmail.com",
            "phone": "(555) 123-4567",
            "age": 25
        })
        
        assert result['success'] is True
        assert result['final_state'] == 'fully_valid'
        assert result['data']['validation_status'] == 'fully_valid'
        assert result['data']['validation_message'] == "All validations passed"
        assert result['data']['validation_score'] == 1.0
        assert len(result['data']['validation_errors']) == 0
        
        # Check formatted phone was created
        assert result['data']['phone_formatted'] == '(555) 123-4567'
        
        # Verify all validations passed
        validations = result['data']['validations']
        assert validations['email']['valid'] is True
        assert validations['phone']['valid'] is True
        assert validations['age']['valid'] is True
    
    def test_invalid_email_domain(self, fsm):
        """Test with invalid email domain."""
        result = fsm.process({
            "email": "user@invalid.com",
            "phone": "5551234567",
            "age": 30
        })
        
        assert result['success'] is True
        assert result['final_state'] == 'partially_valid'
        assert result['data']['validation_status'] == 'partially_valid'
        assert result['data']['validation_message'] == "2/3 validations passed"
        assert result['data']['validation_score'] == 2/3
        
        # Check email validation failed
        assert result['data']['validations']['email']['valid'] is False
        assert any("Domain 'invalid.com' is not in allowed list" in error 
                  for error in result['data']['validation_errors'])
    
    def test_invalid_phone(self, fsm):
        """Test with invalid phone number."""
        result = fsm.process({
            "email": "user@yahoo.com",
            "phone": "123",
            "age": 22
        })
        
        assert result['success'] is True
        assert result['final_state'] == 'partially_valid'
        assert result['data']['validation_status'] == 'partially_valid'
        
        # Check phone validation failed
        assert result['data']['validations']['phone']['valid'] is False
        assert any("Phone must be 10 digits, got 3" in error 
                  for error in result['data']['validation_errors'])
    
    def test_underage(self, fsm):
        """Test with underage user."""
        result = fsm.process({
            "email": "teen@gmail.com",
            "phone": "5551234567",
            "age": 16
        })
        
        assert result['success'] is True
        assert result['final_state'] == 'partially_valid'
        assert result['data']['validation_status'] == 'partially_valid'
        
        # Check age validation failed
        assert result['data']['validations']['age']['valid'] is False
        assert any("Must be 18 or older, got 16" in error 
                  for error in result['data']['validation_errors'])
    
    def test_multiple_issues(self, fsm):
        """Test with multiple validation issues."""
        result = fsm.process({
            "email": "not-an-email",
            "phone": "abc",
            "age": -5
        })
        
        assert result['success'] is True
        assert result['final_state'] == 'invalid'
        assert result['data']['validation_status'] == 'invalid'
        assert result['data']['validation_message'] == "All validations failed"
        assert result['data']['validation_score'] == 0.0
        
        # Check all validations failed
        validations = result['data']['validations']
        assert validations['email']['valid'] is False
        assert validations['phone']['valid'] is False
        assert validations['age']['valid'] is False
        
        # Should have multiple errors
        assert len(result['data']['validation_errors']) >= 3
    
    def test_missing_fields(self, fsm):
        """Test with missing required fields."""
        result = fsm.process({})
        
        assert result['success'] is True
        assert result['final_state'] == 'invalid'
        assert result['data']['validation_status'] == 'invalid'
        assert result['data']['validation_message'] == "All validations failed"
        
        # Check all validations failed due to missing data
        validations = result['data']['validations']
        assert validations['email']['valid'] is False
        assert validations['phone']['valid'] is False
        assert validations['age']['valid'] is False
        
        # Check specific error messages
        assert any("Email is required" in error for error in result['data']['validation_errors'])
        assert any("Phone number is required" in error for error in result['data']['validation_errors'])
        assert any("Age is required" in error for error in result['data']['validation_errors'])
    
    def test_validation_path(self, fsm):
        """Test that FSM follows the correct path through states."""
        result = fsm.process({
            "email": "test@gmail.com",
            "phone": "5551234567",
            "age": 25
        })
        
        expected_path = [
            'input',
            'normalize',
            'validate_email',
            'validate_phone',
            'validate_age',
            'summarize',
            'fully_valid'
        ]
        assert result['path'] == expected_path
    
    def test_partial_validation_path(self, fsm):
        """Test path when some validations fail."""
        result = fsm.process({
            "email": "bad-email",
            "phone": "5551234567",
            "age": 25
        })
        
        # Should still go through all validations but end at partially_valid
        expected_path = [
            'input',
            'normalize',
            'validate_email',
            'validate_phone',
            'validate_age',
            'summarize',
            'partially_valid'
        ]
        assert result['path'] == expected_path
    
    def test_data_normalization(self, fsm):
        """Test that data is properly normalized."""
        result = fsm.process({
            "email": "  JOHN@GMAIL.COM  ",
            "phone": "(555) 123-4567",
            "age": 30
        })
        
        assert result['success'] is True
        # Email should be normalized to lowercase and trimmed
        assert result['data']['email'] == 'john@gmail.com'
        # Phone should have non-digits removed
        assert result['data']['phone'] == '5551234567'
    
    def test_extra_fields_preserved(self, fsm):
        """Test that extra fields are preserved through the pipeline."""
        result = fsm.process({
            "email": "user@gmail.com",
            "phone": "5551234567",
            "age": 25,
            "name": "John Doe",
            "extra": "preserved"
        })
        
        assert result['success'] is True
        assert result['data']['name'] == "John Doe"
        assert result['data']['extra'] == "preserved"
    
    def test_age_boundary_values(self, fsm):
        """Test age validation with boundary values."""
        # Test minimum valid age (18)
        result = fsm.process({
            "email": "user@gmail.com",
            "phone": "5551234567",
            "age": 18
        })
        assert result['data']['validations']['age']['valid'] is True
        
        # Test just below minimum (17)
        result = fsm.process({
            "email": "user@gmail.com",
            "phone": "5551234567",
            "age": 17
        })
        assert result['data']['validations']['age']['valid'] is False
        
        # Test maximum reasonable age (150)
        result = fsm.process({
            "email": "user@gmail.com",
            "phone": "5551234567",
            "age": 150
        })
        assert result['data']['validations']['age']['valid'] is True
        
        # Test just above maximum (151)
        result = fsm.process({
            "email": "user@gmail.com",
            "phone": "5551234567",
            "age": 151
        })
        assert result['data']['validations']['age']['valid'] is False
    
    def test_validation_score_calculation(self, fsm):
        """Test that validation score is correctly calculated."""
        # 0/3 valid
        result = fsm.process({
            "email": "bad",
            "phone": "bad",
            "age": -1
        })
        assert result['data']['validation_score'] == 0.0
        
        # 1/3 valid
        result = fsm.process({
            "email": "user@gmail.com",
            "phone": "bad",
            "age": -1
        })
        assert result['data']['validation_score'] == 1/3
        
        # 2/3 valid
        result = fsm.process({
            "email": "user@gmail.com",
            "phone": "5551234567",
            "age": -1
        })
        assert result['data']['validation_score'] == 2/3
        
        # 3/3 valid
        result = fsm.process({
            "email": "user@gmail.com",
            "phone": "5551234567",
            "age": 25
        })
        assert result['data']['validation_score'] == 1.0