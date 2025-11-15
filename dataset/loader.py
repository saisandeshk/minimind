"""
Dataset loading utilities for various sources and formats.

Currently focuses on HuggingFace datasets with basic support for local files.
"""

from typing import List, Optional
from datasets import load_dataset, Dataset


def load_single_dataset(
    source: str,
    format: str,
    splits: List[str] = ['train'],
    cache_dir: Optional[str] = None, 
    config_name: Optional[str] = None,
) -> Dataset:
    """
    Load a single dataset from various sources.
    
    Args:
        source: Dataset identifier (HF dataset name or local path)
        format: Format type ('huggingface', 'jsonl', 'parquet', 'arrow')
        splits: Which splits to load and concatenate
        cache_dir: Cache directory for downloads
        
    Returns:
        Loaded HuggingFace Dataset
        
    Examples:
        >>> dataset = load_single_dataset("roneneldan/TinyStories", "huggingface")
        >>> print(len(dataset))
    """
    if format == 'huggingface':
        # Prepare common kwargs for load_dataset (do NOT pass trust_remote_code)
        load_kwargs = {}
        if cache_dir:
            load_kwargs['cache_dir'] = cache_dir
        if config_name:
            load_kwargs['name'] = config_name

        try:
            # Concatenate multiple splits if needed
            if len(splits) == 1:
                try:
                    dataset = load_dataset(source, split=splits[0], **load_kwargs)
                except ValueError as e:
                    # Helpful fallback for lighteval/mmlu when config is missing
                    if "Config name is missing" in str(e) and source == "lighteval/mmlu":
                        dataset = load_dataset(source, name="all", split=splits[0], cache_dir=cache_dir)
                    else:
                        raise
            else:
                datasets = []
                for split in splits:
                    try:
                        ds = load_dataset(source, split=split, **load_kwargs)
                    except ValueError as e:
                        if "Config name is missing" in str(e) and source == "lighteval/mmlu":
                            ds = load_dataset(source, name="all", split=split, cache_dir=cache_dir)
                        else:
                            raise
                    datasets.append(ds)

                from datasets import concatenate_datasets
                dataset = concatenate_datasets(datasets)

            return dataset

        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace dataset '{source}': {str(e)}")
    
    elif format == 'jsonl':
        # Load from local JSONL file
        try:
            dataset = load_dataset(
                'json',
                data_files=source,
                split='train',
                cache_dir=cache_dir
            )
            return dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load JSONL file '{source}': {str(e)}")
    
    elif format == 'parquet':
        # Load from Parquet file(s)
        try:
            dataset = load_dataset(
                'parquet',
                data_files=source,
                split='train',
                cache_dir=cache_dir
            )
            return dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load Parquet file '{source}': {str(e)}")
    
    elif format == 'arrow':
        # Load from Arrow format
        try:
            dataset = Dataset.from_file(source)
            return dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load Arrow file '{source}': {str(e)}")
    
    else:
        raise ValueError(f"Unsupported format: {format}. Supported: huggingface, jsonl, parquet, arrow")


def get_dataset_info(source: str, format: str = 'huggingface') -> dict:
    """
    Get information about a dataset without loading it fully.
    
    Args:
        source: Dataset identifier
        format: Dataset format
        
    Returns:
        Dictionary with dataset metadata
    """
    if format == 'huggingface':
        try:
            from datasets import get_dataset_config_names, get_dataset_split_names
            
            info = {
                'source': source,
                'format': format,
                'configs': [],
                'splits': []
            }
            
            try:
                info['configs'] = get_dataset_config_names(source)
            except:
                info['configs'] = ['default']
            
            try:
                info['splits'] = get_dataset_split_names(source)
            except:
                info['splits'] = ['train']
            
            return info
        except Exception as e:
            return {
                'source': source,
                'format': format,
                'error': str(e)
            }
    
    return {'source': source, 'format': format, 'info': 'unavailable'}
