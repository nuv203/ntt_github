# csynthparse.py — parse csynth.xml from Vitis HLS synthesis
# Sourced from github.com/sdrangan/hwdesign (pysilicon/utils/csynthparse.py)

import os
import xml.etree.ElementTree as ET
import pandas as pd

class CsynthParser(object):
    def __init__(self, sol_path=None, report_path=None):
        if (report_path is None) and (sol_path is None):
            raise ValueError("Either sol_path or report_path must be provided.")
        if sol_path is not None:
            report_path = os.path.join(sol_path, 'syn', 'report')
        self.report_xml = os.path.join(report_path, 'csynth.xml')
        if not os.path.exists(self.report_xml):
            raise FileNotFoundError(f"Could not find csynth.xml at {self.report_xml}")

    def get_total_resources(self):
        tree = ET.parse(self.report_xml)
        root = tree.getroot()
        area_estimates = root.find("AreaEstimates")
        if area_estimates is None:
            raise ValueError("No <AreaEstimates> section found in csynth.xml")
        resources_elem = area_estimates.find("Resources")
        self.total_resources = {child.tag: int(child.text) for child in resources_elem}
        avail_elem = area_estimates.find("AvailableResources")
        self.available_resources = {child.tag: int(child.text) for child in avail_elem}

    def get_module_resources(self):
        tree = ET.parse(self.report_xml)
        root = tree.getroot()
        modules_info = {}
        module_info_elem = root.find("ModuleInformation")
        if module_info_elem is None:
            return
        for module in module_info_elem.findall("Module"):
            name_elem = module.find("Name")
            if name_elem is None:
                continue
            module_name = name_elem.text.strip()
            resources_elem = module.find("AreaEstimates/Resources")
            if resources_elem is None:
                continue
            resources = {}
            for child in resources_elem:
                text = child.text.strip()
                try:
                    resources[child.tag] = int(text)
                except ValueError:
                    resources[child.tag] = text
            modules_info[module_name] = resources
        self.module_info = modules_info

    @staticmethod
    def _parse_int_or_nan(text):
        if text is None:
            return float("nan")
        try:
            return int(str(text).strip())
        except (TypeError, ValueError):
            return float("nan")

    @classmethod
    def _parse_latency_range(cls, latency_text):
        if latency_text is None:
            return float("nan"), float("nan")
        parts = [p.strip() for p in str(latency_text).split("~")]
        if len(parts) == 2:
            return cls._parse_int_or_nan(parts[0]), cls._parse_int_or_nan(parts[1])
        value = cls._parse_int_or_nan(latency_text)
        return value, value

    def get_loop_pipeline_info(self):
        tree = ET.parse(self.report_xml)
        root = tree.getroot()
        loop_info = {}
        for module in root.findall(".//ModuleInformation/Module"):
            module_name = module.findtext("Name", default="UnknownModule").strip()
            loop_latency = module.find("PerformanceEstimates/SummaryOfLoopLatency")
            if loop_latency is None:
                continue
            for loop in loop_latency:
                loop_name = loop.findtext("Name", default="UnnamedLoop").strip()
                ii = loop.findtext("PipelineII")
                depth = loop.findtext("PipelineDepth")
                trip_count_min = loop.findtext("TripCount/range/min")
                trip_count_max = loop.findtext("TripCount/range/max")
                latency_min, latency_max = self._parse_latency_range(loop.findtext("Latency"))
                try:
                    ii_val = int(ii)
                except (TypeError, ValueError):
                    ii_val = ii
                try:
                    depth_val = int(depth)
                except (TypeError, ValueError):
                    depth_val = depth
                loop_info[f"{module_name}:{loop_name}"] = {
                    "PipelineII": ii_val,
                    "PipelineDepth": depth_val,
                    "TripCountMin": self._parse_int_or_nan(trip_count_min),
                    "TripCountMax": self._parse_int_or_nan(trip_count_max),
                    "LatencyMin": latency_min,
                    "LatencyMax": latency_max,
                }
        self.loop_df = pd.DataFrame.from_dict(loop_info, orient="index")
        nullable_int_cols = ["PipelineII", "TripCountMin", "TripCountMax", "LatencyMin", "LatencyMax"]
        for col in nullable_int_cols:
            if col in self.loop_df.columns:
                self.loop_df[col] = pd.to_numeric(self.loop_df[col], errors="coerce").astype("Int64")
        if "PipelineDepth" in self.loop_df.columns:
            numeric_depth = pd.to_numeric(self.loop_df["PipelineDepth"], errors="coerce")
            if not numeric_depth.isna().any():
                self.loop_df["PipelineDepth"] = numeric_depth.astype("Int64")

    def get_resources(self):
        self.get_total_resources()
        self.get_module_resources()
        res_types = self.available_resources.keys()
        data = {}
        for mod_name, mod_dict in self.module_info.items():
            m = {res: mod_dict.get(res, 0) for res in res_types}
            data[mod_name] = m
        data['Total'] = self.total_resources
        data['Available'] = self.available_resources
        self.res_df = pd.DataFrame.from_dict(data, orient="index")
