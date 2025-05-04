class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3

class Track:
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative
        
        # Appearance features management
        self.features = []
        self.feature_qualities = []
        if feature is not None:
            self.features.append(feature)
            self.feature_qualities.append(1.0)  # Initial quality
            
        self._n_init = n_init
        self._max_age = max_age
        self._max_features = 10  # Maximum number of features to store

    def update(self, kf, detection):
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
            
        if detection.feature is not None:
            # Calculate feature quality based on detection confidence and size
            box = detection.to_tlbr()
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            quality = min(1.0, (getattr(detection, 'confidence', 1.0) * (box_area / 10000)))
            
            # Manage feature storage
            if len(self.features) >= self._max_features:
                # Replace the lowest quality feature
                min_idx = self.feature_qualities.index(min(self.feature_qualities))
                self.features.pop(min_idx)
                self.feature_qualities.pop(min_idx)
                
            self.features.append(detection.feature)
            self.feature_qualities.append(quality)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def get_best_features(self, n=3):
        if not self.features:
            return []
            
        # Get indices of top n features by quality
        indices = sorted(range(len(self.feature_qualities)), 
                        key=lambda i: self.feature_qualities[i], 
                        reverse=True)[:n]
        return [self.features[i] for i in indices]

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted

    def to_tlwh(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1