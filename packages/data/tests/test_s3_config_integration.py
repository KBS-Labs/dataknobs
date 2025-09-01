"""Test S3 backend configuration integration."""

import os
import pytest
from unittest.mock import patch, MagicMock
from dataknobs_config import Config
from dataknobs_data.backends.s3 import SyncS3Database


class TestS3ConfigIntegration:
    """Test S3 backend integration with Config class."""
    
    @patch('boto3.client')
    def test_s3_config_integration(self, mock_boto_client):
        """Test that SyncS3Database can be instantiated via Config."""
        # Mock S3 client
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        mock_s3.head_bucket.return_value = {}  # Bucket exists
        
        # Create configuration
        config = Config()
        config.load({
            "databases": [{
                "name": "s3_storage",
                "class": "dataknobs_data.backends.s3.SyncS3Database",
                "bucket": "my-test-bucket",
                "prefix": "records/prod/",
                "region": "us-west-2",
                "max_workers": 20
            }]
        })
        
        # Get instance through Config
        db = config.get_instance("databases", "s3_storage")
        
        # Verify it's an SyncS3Database instance
        assert isinstance(db, SyncS3Database)
        assert db.bucket == "my-test-bucket"
        assert db.prefix == "records/prod/"
        assert db.region == "us-west-2"
        assert db.max_workers == 20
    
    @patch('boto3.client')
    def test_s3_config_with_environment_variables(self, mock_boto_client, monkeypatch):
        """Test S3 configuration with environment variable substitution."""
        # Mock S3 client
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        mock_s3.head_bucket.return_value = {}  # Bucket exists
        
        # Detect if we're running in Docker container
        if os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER'):
            localstack_host = 'localstack'
        else:
            localstack_host = 'localhost'
        
        # Set environment variables
        monkeypatch.setenv("S3_BUCKET", "env-bucket")
        monkeypatch.setenv("S3_PREFIX", "env-prefix/")
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        monkeypatch.setenv("LOCALSTACK_ENDPOINT", f"http://{localstack_host}:4566")
        
        # Create configuration with environment variables
        config = Config()
        config.load({
            "databases": [{
                "name": "s3_env",
                "class": "dataknobs_data.backends.s3.SyncS3Database",
                "bucket": "${S3_BUCKET}",
                "prefix": "${S3_PREFIX}",
                "region": "${AWS_REGION}",
                "endpoint_url": "${LOCALSTACK_ENDPOINT}"
            }]
        })
        
        # Get instance
        db = config.get_instance("databases", "s3_env")
        
        # Verify environment variables were substituted
        assert db.bucket == "env-bucket"
        assert db.prefix == "env-prefix/"
        assert db.region == "eu-west-1"
        assert db.endpoint_url == f"http://{localstack_host}:4566"
    
    @patch('boto3.client')
    def test_s3_config_with_defaults(self, mock_boto_client, monkeypatch):
        """Test S3 configuration with default values."""
        # Mock S3 client
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        mock_s3.head_bucket.return_value = {}  # Bucket exists
        
        # Remove environment variable if set
        monkeypatch.delenv("S3_BUCKET", raising=False)
        
        # Create configuration with defaults
        config = Config()
        config.load({
            "databases": [{
                "name": "s3_defaults",
                "class": "dataknobs_data.backends.s3.SyncS3Database",
                "bucket": "${S3_BUCKET:default-bucket}",
                "prefix": "${S3_PREFIX:default-prefix/}",
                "region": "${AWS_REGION:us-east-1}"
            }]
        })
        
        # Get instance
        db = config.get_instance("databases", "s3_defaults")
        
        # Verify defaults were used
        assert db.bucket == "default-bucket"
        assert db.prefix == "default-prefix/"
        assert db.region == "us-east-1"
    
    @patch('boto3.client')
    def test_s3_from_config_directly(self, mock_boto_client):
        """Test creating SyncS3Database directly with from_config."""
        # Mock S3 client
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        mock_s3.head_bucket.return_value = {}  # Bucket exists
        
        config_dict = {
            "bucket": "direct-bucket",
            "prefix": "direct/",
            "region": "ap-southeast-1",
            "max_workers": 15,
            "max_retries": 5
        }
        
        db = SyncS3Database.from_config(config_dict)
        
        assert isinstance(db, SyncS3Database)
        assert db.bucket == "direct-bucket"
        assert db.prefix == "direct/"
        assert db.region == "ap-southeast-1"
        assert db.max_workers == 15
        assert db.max_retries == 5
    
    def test_s3_missing_required_config(self):
        """Test that missing required configuration raises error."""
        config = Config()
        config.load({
            "databases": [{
                "name": "s3_invalid",
                "class": "dataknobs_data.backends.s3.SyncS3Database",
                # Missing required 'bucket' parameter
                "prefix": "records/"
            }]
        })
        
        # Should raise ValueError about missing bucket
        with pytest.raises(ValueError, match="bucket name is required"):
            config.get_instance("databases", "s3_invalid")
