# sampler.py
import random
import torch

class EpisodeSampler:
    """Samples episodes for N-way K-shot learning."""
    def __init__(self, meta_data, n_way, k_shot, n_query):
        """
        Initializes the EpisodeSampler.

        Args:
            meta_data (list): A list of tuples, where each tuple is
                              (subset_class_id, [list_of_image_tensors]).
            n_way (int): Number of classes per episode.
            k_shot (int): Number of support examples per class.
            n_query (int): Number of query examples per class.
        """
        if not meta_data:
            raise ValueError("Meta data cannot be empty or None.")
        self.meta_data = meta_data
        self.num_classes_available = len(meta_data)

        if n_way <= 0 or n_way > self.num_classes_available:
            raise ValueError(f"Invalid n_way ({n_way}). Must be > 0 and <= available classes ({self.num_classes_available}).")
        if k_shot <= 0:
             raise ValueError(f"Invalid k_shot ({k_shot}). Must be > 0.")
        if n_query <= 0:
             raise ValueError(f"Invalid n_query ({n_query}). Must be > 0.")

        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query

        # Pre-calculate minimum required samples per class and check availability
        self.min_samples_needed = self.k_shot + self.n_query
        self.class_indices_with_sufficient_samples = [
            i for i, (_, images) in enumerate(self.meta_data)
            if len(images) >= self.min_samples_needed
        ]
        self.class_indices_needing_replacement = [
             i for i, (_, images) in enumerate(self.meta_data)
             if 0 < len(images) < self.min_samples_needed
        ]

        num_sufficient = len(self.class_indices_with_sufficient_samples)
        num_insufficient = len(self.class_indices_needing_replacement)

        if num_sufficient < self.n_way:
             print(f"Warning: Only {num_sufficient} classes have >= {self.min_samples_needed} samples.")
             print(f"         {num_insufficient} classes have < {self.min_samples_needed} samples but > 0.")
             print(f"         Sampling N_WAY={self.n_way} might rely heavily on replacement or fail if not enough classes.")
             if num_sufficient + num_insufficient < self.n_way:
                 raise ValueError(f"Cannot sample {self.n_way} ways. Only {num_sufficient + num_insufficient} classes have any images.")
        # Store the actual indices used to map back to meta_data
        self.usable_class_indices = self.class_indices_with_sufficient_samples + self.class_indices_needing_replacement


    def sample(self):
        """Samples a single N-way K-shot episode."""
        support_imgs, support_lbls, query_imgs, query_lbls = [], [], [], []

        # Sample N_WAY classes from the usable indices
        try:
            # Prioritize classes with enough samples if possible, but allow insufficient ones
            sampled_meta_data_indices = random.sample(self.usable_class_indices, self.n_way)
        except ValueError:
            # This should be caught by the __init__ check, but as a safeguard
            raise ValueError(f"Cannot sample {self.n_way} ways from {len(self.usable_class_indices)} usable classes.")

        for local_label, meta_data_idx in enumerate(sampled_meta_data_indices):
            # Get the actual subset_class_id (unused here) and image list
            _, images = self.meta_data[meta_data_idx]
            n_available = len(images)

            # Determine if replacement is needed for this specific class
            use_replacement = n_available < self.min_samples_needed

            if use_replacement:
                # Sample with replacement if needed
                selected_indices = random.choices(range(n_available), k=self.min_samples_needed)
            else:
                # Sample without replacement if enough images are available
                selected_indices = random.sample(range(n_available), self.min_samples_needed)

            # Assign images to support and query sets
            support_imgs.extend([images[i] for i in selected_indices[:self.k_shot]])
            support_lbls.extend([local_label] * self.k_shot) # Use 0 to N_WAY-1 labels
            query_imgs.extend([images[i] for i in selected_indices[self.k_shot:]])
            query_lbls.extend([local_label] * self.n_query)

        # Stack tensors
        support_imgs = torch.stack(support_imgs)
        support_lbls = torch.LongTensor(support_lbls)
        query_imgs = torch.stack(query_imgs)
        query_lbls = torch.LongTensor(query_lbls)

        # Shuffle the query set (support set order doesn't matter for prototype calculation)
        perm = torch.randperm(len(query_lbls))
        query_imgs = query_imgs[perm]
        query_lbls = query_lbls[perm]

        return support_imgs, support_lbls, query_imgs, query_lbls