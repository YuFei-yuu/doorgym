"""Scoring only. Never commands the expert or authorizes a physical latch."""
import math

class SuccessMonitor:
    def __init__(self):
        self.max_abs_angle = 0.
        self.opened_at = None
        self.legal_relock_at = None
        self.legal_relock_evidence = None
        self.final_relocked = False

    def authorize_relock(self, t, measured_angle, gate):
        # Called ONLY after the unchanged authorize_latch returned a request.
        self.legal_relock_at = float(t)
        self.legal_relock_evidence = dict(measured_angle=float(measured_angle), gate=gate)

    def observe(self, t, measured_angle, relocked):
        self.max_abs_angle = max(self.max_abs_angle, abs(float(measured_angle)))
        if self.opened_at is None and self.max_abs_angle >= math.radians(80.):
            self.opened_at = float(t)
        self.final_relocked = bool(relocked)

    def result(self):
        opened = self.opened_at is not None
        legal = self.legal_relock_at is not None and self.final_relocked
        return dict(success=bool(opened and legal), opened_80=opened,
                    max_abs_open_deg=math.degrees(self.max_abs_angle),
                    opened_80_time_s=self.opened_at, legal_relock_time_s=self.legal_relock_at,
                    final_legal_relocked=legal, legal_relock_evidence=self.legal_relock_evidence,
                    criterion='same episode: max(abs(measured hinge angle)) >= 80 deg AND final legal relock',
                    failures=(["did_not_open_80"] if not opened else []) +
                             (["not_final_legally_relocked"] if not legal else []))
