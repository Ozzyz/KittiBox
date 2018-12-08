#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <numeric>
#include <strings.h>
#include <assert.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>

#include "eval_structs.h"
using namespace std;

/*=======================================================================
STATIC EVALUATION PARAMETERS
=======================================================================*/

// easy, moderate and hard evaluation level
enum DIFFICULTY
{
  EASY = 0,
  MODERATE = 1,
  HARD = 2
};

// evaluation parameter
const int32_t MIN_HEIGHT[3] = {40, 40, 40};                       // minimum height for evaluated groundtruth/detections
const int32_t MAX_OCCLUSION[3] = {1000, 1000, 2000};       // maximum occlusion level of the groundtruth used for evaluation
const double MAX_TRUNCATION[3] = {1000, 1000, 1000}; // maximum truncation level of the groundtruth used for evaluation

// evaluated object classes
enum CLASSES
{
  CAR = 0,
  PERSON = 1,
  BIKE = 2,
  TRAFFIC_LIGHT = 3,
  TRAFFIC_SIGN = 4,
  TRUCK = 5
};

// parameters varying per class
vector<string> CLASS_NAMES;
const double MIN_OVERLAP[6] = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2}; // the minimum overlap required for evaluation

// no. of recall steps that should be evaluated (discretized)
const double N_SAMPLE_PTS = 41;

// initialize class names
void initGlobals()
{
  CLASS_NAMES.push_back("car");
  CLASS_NAMES.push_back("person");
  CLASS_NAMES.push_back("bike");
  CLASS_NAMES.push_back("traffic_light");
  CLASS_NAMES.push_back("traffic_sign");
  CLASS_NAMES.push_back("truck");
}

/*=======================================================================
FUNCTIONS TO LOAD DETECTION AND GROUND TRUTH DATA ONCE, SAVE RESULTS
=======================================================================*/
vector<tDetection> loadDetections(string file_name, bool &compute_aos, bool &eval_car, bool &eval_person, bool &eval_bike, bool &eval_trafficlight, bool &eval_trafficsign, bool &eval_truck, bool &success)
{
  /* Loads the detections from the model from the given filename.
     Returns a vector of detections where each element is of type tDetection struct.
   */
  //cout << "Loading detections filename: " << file_name <<  endl;
  // holds all detections (ignored detections are indicated by an index vector
  vector<tDetection> detections;
  FILE *fp = fopen(file_name.c_str(), "r");
  if (!fp)
  {
    cout << "ERROR: Failed to load file: " << file_name << endl;
    success = false;
    return detections;
  }
  //cout << "Loading detections from " << file_name << endl;
  while (!feof(fp))
  {
    tDetection d;
    double trash;
    char str[255];
    int num_args = fscanf(fp, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                          str, &trash, &trash, &d.box.alpha,
                          &d.box.x1, &d.box.y1, &d.box.x2, &d.box.y2,
                          &trash, &trash, &trash, &trash,
                          &trash, &trash, &trash, &d.thresh);
    if (num_args == 16)
    {
      d.box.type = str;
      detections.push_back(d);
      //cout << "\tloadDetections: Loaded values " << "Class: " << str << ", X1,Y1,X2,Y2: " << d.box.x1 << " "<< d.box.y1 << " " << d.box.x2 << " "<< d.box.y2 << endl;

      // orientation=-10 is invalid, AOS is not evaluated if at least one orientation is invalid
      if (d.box.alpha == -10)
        compute_aos = false;
      // a class is only evaluated if it is detected at least once
      if (!eval_person && !strcasecmp(d.box.type.c_str(), "person"))
        eval_person = true;
      if (!eval_bike && !strcasecmp(d.box.type.c_str(), "bike"))
        eval_bike = true;
      if (!eval_trafficlight && !strcasecmp(d.box.type.c_str(), "traffic_light"))
        eval_trafficlight = true;
      if (!eval_trafficsign && !strcasecmp(d.box.type.c_str(), "traffic_sign"))
        eval_trafficsign = true;
      if (!eval_truck && !strcasecmp(d.box.type.c_str(), "truck"))
        eval_truck = true;
      if (!eval_car && !strcasecmp(d.box.type.c_str(), "car"))
        eval_car = true;
    }
    else
    {
      //cout << "\t\t\t loadDetections: Could not load detections from fscanf of file " << file_name << ", are you sure it is formatted correctly? (16 values per line)" << endl;
      //cout << "Note that classes other than car, pedestrian and cyclists are ignored" << endl;
      //cout << "FAIL! Values read: " << str << ", alpha, dbox (x1, y1, x2, y2), tresh " << d.box.alpha << ", " << d.box.x1 << d.box.y1 << d.box.x2 << d.box.y2 << d.thresh << endl;
    }
  }
  fclose(fp);
  success = true;
  return detections;
}

vector<tGroundtruth> loadGroundtruth(string file_name, bool &success)
{
  /* Attempts to load all ground truths (kitti-formatted bboxes) from the given filename
     and returns a vector of groundtruths where each element is a tGroundTruth struct.
     Success is set to false if the function fails to load the file given by file_name.
  */

  // holds all ground truth (ignored ground truth is indicated by an index vector
  vector<tGroundtruth> groundtruth;
  //cout << "Loading ground truth, filename: " << file_name << endl;
  
  FILE *fp = fopen(file_name.c_str(), "r");
  if (!fp)
  {
    cout << "loadGroundtruth: Failed to load filename " << file_name << endl;
    success = false;
    return groundtruth;
  }
  while (!feof(fp))
  {
    tGroundtruth g;
    double trash;
    char str[255];
    // The %s argument of fscanf expects a string without whitespaces.
    int num_args = fscanf(fp, "%s %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                          str, &g.truncation, &g.occlusion, &g.box.alpha,
                          &g.box.x1, &g.box.y1, &g.box.x2, &g.box.y2,
                          &trash, &trash, &trash, &trash,
                          &trash, &trash, &trash);
    if (num_args == 15)
    {
      //cout << "Values read: " << str << ",alpha,  dbox (x1, y1, x2, y2) " << g.box.alpha << ", " << g.box.x1 << ", " << g.box.y1 << ", " << g.box.x2 << ", " << g.box.y2 << endl;
      g.box.type = str;
      //cout << "\tloadGroundTruths: Loaded values " << "Class: " << str << ", X1,Y1,X2,Y2: " << g.box.x1 << " "<< g.box.y1 << " " << g.box.x2 << " "<< g.box.y2 << endl;
      groundtruth.push_back(g);
    }
    /*else
    {
      cout << "Could not load ground truths in fscanf of file " << file_name << ", are you sure it is formatted correctly? (15 values per line) " << endl;
    }*/
  }
  fclose(fp);
  //cout << "Successfully closed file and loaded ground truth " << endl;
  //cout << "groundtruth length: " << groundtruth.size() << endl;
  success = true;
  return groundtruth;
}

void saveStats(const vector<double> &precision, const vector<double> &aos, FILE *fp_det, FILE *fp_ori)
{
  /* Writes the precision vector to fp_det, and aos-vector to fp_ori if they are non-empty */
  //cout << "Saving stats to file " << endl;
  // save precision to file
  if (precision.empty())
  {
    cout << "Precision vector empty -- exiting. " << endl;
    return;
  }
  //cout << "Writing precision elements: ";
  cout << "\t";
  for (int32_t i = 0; i < precision.size(); i++)
  {
    cout << "," << precision[i];
    fprintf(fp_det, "%f ", precision[i]);
  }
  cout << endl;
  fprintf(fp_det, "\n");

  // save orientation similarity, only if there were no invalid orientation entries in submission (alpha=-10)
  if (aos.empty())
    return;
  for (int32_t i = 0; i < aos.size(); i++)
  {
    cout << "Writing " << aos[i] << " to fp_ori" << endl;
    fprintf(fp_ori, "%f ", aos[i]);
  }
  fprintf(fp_ori, "\n");
}

/*=======================================================================
EVALUATION HELPER FUNCTIONS
=======================================================================*/

// criterion defines whether the overlap is computed with respect to both areas (ground truth and detection)
// or with respect to box a or b (detection and "dontcare" areas)
inline double boxoverlap(tBox a, tBox b, int32_t criterion = -1)
{
  // overlap is invalid in the beginning
  double o = -1;

  // get overlapping area
  double x1 = max(a.x1, b.x1);
  double y1 = max(a.y1, b.y1);
  double x2 = min(a.x2, b.x2);
  double y2 = min(a.y2, b.y2);

  // compute width and height of overlapping area
  double w = x2 - x1;
  double h = y2 - y1;

  // set invalid entries to 0 overlap
  if (w <= 0 || h <= 0)
    return 0;

  // get overlapping areas
  double inter = w * h;
  double a_area = (a.x2 - a.x1) * (a.y2 - a.y1);
  double b_area = (b.x2 - b.x1) * (b.y2 - b.y1);

  // intersection over union overlap depending on users choice
  if (criterion == -1) // union
    o = inter / (a_area + b_area - inter);
  else if (criterion == 0) // bbox_a
    o = inter / a_area;
  else if (criterion == 1) // bbox_b
    o = inter / b_area;

  //cout << "Calculating overlap between box 1: (" << a.x1 << ", " << a.y1 << ", " << a.x2 << ", " << a.y2;
  //cout << ") and box 2: (" << b.x1 << ", " << b.y1 << ", " << b.x2 << ", " << b.y2 << endl;
  //cout << "Overlap is : " << o << endl;
  // overlap
  return o;
}

vector<double> getThresholds(vector<double> &v, double n_groundtruth)
{

  //cout << "\tComputed threshold" << endl;
  // holds scores needed to compute N_SAMPLE_PTS recall values
  vector<double> t;

  // sort scores in descending order
  // (highest score is assumed to give best/most confident detections)
  sort(v.begin(), v.end(), greater<double>());

  // get scores for linearly spaced recall
  double current_recall = 0;
  for (int32_t i = 0; i < v.size(); i++)
  {

    // check if right-hand-side recall with respect to current recall is close than left-hand-side one
    // in this case, skip the current detection score
    double l_recall, r_recall, recall;
    l_recall = (double)(i + 1) / n_groundtruth;
    if (i < (v.size() - 1))
    {
      r_recall = (double)(i + 2) / n_groundtruth;
    }
    else
    {
      r_recall = l_recall;
    }
    if ((r_recall - current_recall) < (current_recall - l_recall) && i < (v.size() - 1))
    {
      continue;
    }
    // left recall is the best approximation, so use this and goto next recall step for approximation
    recall = l_recall;

    // the next recall step was reached
    t.push_back(v[i]);
    current_recall += 1.0 / (N_SAMPLE_PTS - 1.0);
  }
  return t;
}

void cleanData(CLASSES current_class, const vector<tGroundtruth> &gt, const vector<tDetection> &det, vector<int32_t> &ignored_gt, vector<tGroundtruth> &dc, vector<int32_t> &ignored_det, int32_t &n_gt, DIFFICULTY difficulty)
{
  /* Iterates through the ground truths (gt) and ignores all ground truths that are too occluded (defined by MAX_OCCLUSION[difficulty], MAX_TRUNCATION and MIN_HEIGHT)
     Then, it creates a vector (ignored_gt) that is 0 on index i if gt[i] should not be ignored. Else, it is ignored.
   */
  //cout << "\t\tClean data called, current class : " << CLASS_NAMES[current_class] << endl;
  // extract ground truth bounding boxes for current evaluation class

  //cout << "\t\tcleanData: Computing TP, FP and FN" << endl;
  //cout << "\t\tsize of gt, det, ignored_gt, dc, ignored_det: " << gt.size() << ", " << det.size() << ", " << ignored_gt.size() << ", " << dc.size() << ", " << ignored_det.size() << endl;
  
  for (int32_t i = 0; i < gt.size(); i++)
  {
    // only bounding boxes with a minimum height are used for evaluation
    double height = gt[i].box.y2 - gt[i].box.y1;

    // neighboring classes are ignored ("van" for "car" and "person_sitting" for "pedestrian")
    // (lower/upper cases are ignored)
    int32_t valid_class;

    // all classes without a neighboring class
    if (!strcasecmp(gt[i].box.type.c_str(), CLASS_NAMES[current_class].c_str())){
      valid_class = 1;
    }
    // classes not used for evaluation
    else{
      valid_class = -1;
    }
    // ground truth is ignored, if occlusion, truncation exceeds the difficulty or ground truth is too small
    // (doesn't count as FN nor TP, although detections may be assigned)
    bool ignore = false;
    if (gt[i].occlusion > MAX_OCCLUSION[difficulty] || gt[i].truncation > MAX_TRUNCATION[difficulty] || height < MIN_HEIGHT[difficulty]){
      ignore = true;
    }
    // set ignored vector for ground truth
    // current class and not ignored (total no. of ground truth is detected for recall denominator)
    if (valid_class == 1 && !ignore){
      ignored_gt.push_back(0);
      n_gt++;
    }
    // neighboring class, or current class but ignored
    else if (valid_class == 0 || (ignore && valid_class == 1)){
      ignored_gt.push_back(1);
    }
    // all other classes which are FN in the evaluation
    else{
      ignored_gt.push_back(-1);
    }
  }

  // extract dontcare areas
  for (int32_t i = 0; i < gt.size(); i++)
    if (!strcasecmp("DontCare", gt[i].box.type.c_str()))
      dc.push_back(gt[i]);

  // extract detections bounding boxes of the current class
  for (int32_t i = 0; i < det.size(); i++)
  {

    // neighboring classes are not evaluated
    int32_t valid_class;
    if (!strcasecmp(det[i].box.type.c_str(), CLASS_NAMES[current_class].c_str())){
      valid_class = 1;
    }
    else{
      valid_class = -1;
    }
    // set ignored vector for detections
    if (valid_class == 1){
      ignored_det.push_back(0);
    }
    else{
      ignored_det.push_back(-1);
    }
  }
  //cout << "\t\tcleanData finished" << endl;
}

tPrData computeStatistics(CLASSES current_class, const vector<tGroundtruth> &gt, const vector<tDetection> &det, const vector<tGroundtruth> &dc, const vector<int32_t> &ignored_gt, const vector<int32_t> &ignored_det, bool compute_fp, bool compute_aos = false, double thresh = 0, bool debug = false)
{
  //cout << "Computing statistics of class " << CLASS_NAMES[current_class] << endl;
  tPrData stat = tPrData();
  const double NO_DETECTION = -10000000;
  vector<double> delta;            // holds angular difference for TPs (needed for AOS evaluation)
  vector<bool> assigned_detection; // holds wether a detection was assigned to a valid or ignored ground truth
  assigned_detection.assign(det.size(), false);
  //cout << "\tAssigning assigned_detection size: " << det.size() << endl;
  vector<bool> ignored_threshold;
  ignored_threshold.assign(det.size(), false); // holds detections with a threshold lower than thresh if FP are computed

  // detections with a low score are ignored for computing precision (needs FP)
  if (compute_fp)
  {
    for (int32_t i = 0; i < det.size(); i++)
    {
      // cout << "Thresh of det: " << det[i].thresh << " thresh of computestatistics: " << thresh << endl;
      if (det[i].thresh < thresh)
        ignored_threshold[i] = true;
    }
  }
  // evaluate all ground truth boxes
  //cout << "\tcomputeStatistics: Entering iteration over all gts in vector (" << gt.size() << ")" << endl;
  for (int32_t i = 0; i < gt.size(); i++)
  {

    // this ground truth is not of the current or a neighboring class and therefore ignored
    if (ignored_gt[i] == -1)
      continue;

    /*=======================================================================
    find candidates (overlap with ground truth > 0.5) (logical len(det))
    =======================================================================*/
    int32_t det_idx = -1;
    double valid_detection = NO_DETECTION;
    double max_overlap = 0;

    // search for a possible detection
    bool assigned_ignored_det = false;
    for (int32_t j = 0; j < det.size(); j++)
    {

      // detections not of the current class, already assigned or with a low threshold are ignored
      if (ignored_det[j] == -1)
      {
        //cout << "\tSkipping " << j << "  because of ignored det == -1" << endl;
        continue;
      }
      if (assigned_detection[j])
      {
        //cout << "\tSkipping " << j << " because of already assigned detection" << endl;
        continue;
      }
      if (ignored_threshold[j])
      {
        //cout << "Skipping " << j << " because of ignored threshold" << endl;
        continue;
      }

      // find the maximum score for the candidates and get idx of respective detection
      double overlap = boxoverlap(det[j].box, gt[i].box);
      // for computing recall thresholds, the candidate with highest score is considered
      if (!compute_fp && overlap > MIN_OVERLAP[current_class] && det[j].thresh > valid_detection)
      {
        det_idx = j;
        valid_detection = det[j].thresh;
      }

      // for computing pr curve values, the candidate with the greatest overlap is considered
      // if the greatest overlap is an ignored detection (min_height), the overlapping detection is used
      else if (compute_fp && overlap > MIN_OVERLAP[current_class] && (overlap > max_overlap || assigned_ignored_det) && ignored_det[j] == 0)
      {
        max_overlap = overlap;
        det_idx = j;
        valid_detection = 1;
        assigned_ignored_det = false;
      }
      else if (compute_fp && overlap > MIN_OVERLAP[current_class] && valid_detection == NO_DETECTION && ignored_det[j] == 1)
      {
        det_idx = j;
        valid_detection = 1;
        assigned_ignored_det = true;
      }
    }

    /*=======================================================================
    compute TP, FP and FN
    =======================================================================*/

    // nothing was assigned to this valid ground truth
    if (valid_detection == NO_DETECTION && ignored_gt[i] == 0)
      stat.fn++;

    // only evaluate valid ground truth <=> detection assignments (considering difficulty level)
    else if (valid_detection != NO_DETECTION && (ignored_gt[i] == 1 || ignored_det[det_idx] == 1))
      assigned_detection[det_idx] = true;

    // found a valid true positive
    else if (valid_detection != NO_DETECTION)
    {

      // write highest score to threshold vector
      stat.tp++;
      stat.v.push_back(det[det_idx].thresh);

      // compute angular difference of detection and ground truth if valid detection orientation was provided
      if (compute_aos)
        delta.push_back(gt[i].box.alpha - det[det_idx].box.alpha);

      // clean up
      assigned_detection[det_idx] = true;
    }
  }

  // if FP are requested, consider stuff area
  if (compute_fp)
  {

    //cout << "\tcomputeStatistics: Computing fp line 410" << endl;
    // count fp
    for (int32_t i = 0; i < det.size(); i++)
    {

      // count false positives if required (height smaller than required is ignored (ignored_det==1)
      if (!(assigned_detection[i] || ignored_det[i] == -1 || ignored_det[i] == 1 || ignored_threshold[i]))
        stat.fp++;
    }

    // do not consider detections overlapping with stuff area
    int32_t nstuff = 0;
    for (int32_t i = 0; i < dc.size(); i++)
    {
      for (int32_t j = 0; j < det.size(); j++)
      {

        // detections not of the current class, already assigned, with a low threshold or a low minimum height are ignored
        if (assigned_detection[j])
          continue;
        if (ignored_det[j] == -1 || ignored_det[j] == 1)
          continue;
        if (ignored_threshold[j])
          continue;

        // compute overlap and assign to stuff area, if overlap exceeds class specific value
        double overlap = boxoverlap(det[j].box, dc[i].box, 0);
        if (overlap > MIN_OVERLAP[current_class])
        {
          cout << "\t Overlap: " << overlap << endl;
          assigned_detection[j] = true;
          nstuff++;
        }
      }
    }

    // FP = no. of all not to ground truth assigned detections - detections assigned to stuff areas
    stat.fp -= nstuff;

    // if all orientation values are valid, the AOS is computed
    if (compute_aos)
    {
      vector<double> tmp;

      // FP have a similarity of 0, for all TP compute AOS
      tmp.assign(stat.fp, 0);
      for (int32_t i = 0; i < delta.size(); i++)
        tmp.push_back((1.0 + cos(delta[i])) / 2.0);

      // be sure, that all orientation deltas are computed
      assert(tmp.size() == stat.fp + stat.tp);
      assert(delta.size() == stat.tp);

      // get the mean orientation similarity for this image
      if (stat.tp > 0 || stat.fp > 0)
        stat.similarity = accumulate(tmp.begin(), tmp.end(), 0.0);

      // there was neither a FP nor a TP, so the similarity is ignored in the evaluation
      else
        stat.similarity = -1;
    }
  }
  //cout << "\tCall to computeStatistics finished" << endl;
  //cout << "\tStat values: (FN, TP, similarity): " << stat.fn << ", " << stat.tp << ", " << stat.similarity <<  endl;
  return stat;
}

/*=======================================================================
EVALUATE CLASS-WISE
=======================================================================*/
// TODO: Change the variable N_TESTIMAGES to the correct number for bdd100k
bool eval_class(FILE *fp_det, FILE *fp_ori, CLASSES current_class, const vector<vector<tGroundtruth>> &groundtruth, const vector<vector<tDetection>> &detections, bool compute_aos, vector<double> &precision, vector<double> &aos, DIFFICULTY difficulty)
{
  //cout << "\t\tEntering eval class - class: " << CLASS_NAMES[current_class] << endl;
  // init
  int32_t n_gt = 0;                                // total no. of gt (denominator of recall)
  vector<double> v, thresholds;                    // detection scores, evaluated for recall discretization
  vector<vector<int32_t>> ignored_gt, ignored_det; // index of ignored gt detection for current class/difficulty
  vector<vector<tGroundtruth>> dontcare;           // index of dontcare areas, included in ground truth
  //cout << "\t\tSize of gts and detections: " << groundtruth.size() << ", " << detections.size() << endl;
  // for all test images do
  for (int32_t i = 0; i < detections.size(); i++)
  {
    if(groundtruth[i].size() > 1000 || detections[i].size() > 1000){
        continue; // TODO: Check if this is necessary 
    }
    // holds ignored ground truth, ignored detections and dontcare areas for current frame
    vector<int32_t> i_gt, i_det;
    vector<tGroundtruth> dc;

    // only evaluate objects of current class and ignore occluded, truncated objects
    cleanData(current_class, groundtruth[i], detections[i], i_gt, dc, i_det, n_gt, difficulty);

    ignored_gt.push_back(i_gt);
    ignored_det.push_back(i_det);
    dontcare.push_back(dc);
    // compute statistics to get recall values
    tPrData pr_tmp = tPrData();
    pr_tmp = computeStatistics(current_class, groundtruth[i], detections[i], dc, i_gt, i_det, false);
    //cout << "\t\tCall to computestatistics returned" << endl;
    // add detection scores to vector over all images
    for (int32_t j = 0; j < pr_tmp.v.size(); j++){
      v.push_back(pr_tmp.v[j]);
    }
    //cout << "\t\t\tFinished pushing back tmp vector scores" << endl;
  }
  //cout << "\t\t Getting thresholds for recall discretiozation" << endl;
  // get scores that must be evaluated for recall discretization
  thresholds = getThresholds(v, n_gt);
  //cout << "\t\tFinished call to getthresholds" << endl;
  // compute TP,FP,FN for relevant scores
  vector<tPrData> pr;
  pr.assign(thresholds.size(), tPrData());

  for (int32_t i = 0; i < detections.size(); i++)
  {

    // for all scores/recall thresholds do:
    for (int32_t t = 0; t < thresholds.size(); t++)
    {
      tPrData tmp = tPrData();
      tmp = computeStatistics(current_class, groundtruth[i], detections[i], dontcare[i],
                              ignored_gt[i], ignored_det[i], true, compute_aos, thresholds[t], t == 38); // TODO: Find out why the fuck this is called with t==38

      // add no. of TP, FP, FN, AOS for current frame to total evaluation for current threshold
      pr[t].tp += tmp.tp;
      pr[t].fp += tmp.fp;
      pr[t].fn += tmp.fn;
      if (tmp.similarity != -1)
        pr[t].similarity += tmp.similarity;
    }
  }

  // compute recall, precision and AOS
  //cout << "\t\t Computing recall, precision and AOS " << endl;
  vector<double> recall;
  precision.assign(N_SAMPLE_PTS, 0);
  if (compute_aos)
    aos.assign(N_SAMPLE_PTS, 0);
  double r = 0;
  //cout << "\t\tSize of thresholds in eval_class: " << thresholds.size() << endl;
  for (int32_t i = 0; i < thresholds.size(); i++)
  {
    r = pr[i].tp / (double)(pr[i].tp + pr[i].fn);
    recall.push_back(r);
    precision[i] = pr[i].tp / (double)(pr[i].tp + pr[i].fp);
    //cout << "Calculating precision: "
    //     << "TP, FP, Recall, Precision: " << pr[i].tp << ", " << pr[i].fp << ", " << r << ", " << precision[i] << endl;
    if (compute_aos)
      aos[i] = pr[i].similarity / (double)(pr[i].tp + pr[i].fp);
  }
  //cout << "Sucessfully computed recall, precision and AOS" << endl;

  // filter precision and AOS using max_{i..end}(precision)
  for (int32_t i = 0; i < thresholds.size(); i++)
  {
    precision[i] = *max_element(precision.begin() + i, precision.end());
    //cout << "eval_class(): Calculating precision[" << i << "] to be " << precision[i] << endl;
    if (compute_aos)
      aos[i] = *max_element(aos.begin() + i, aos.end());
  }
  //cout << "\tSucessfully filtered precision and AOS" << endl;

  // save statisics and finish with success
  saveStats(precision, aos, fp_det, fp_ori);
  //cout << "\tSucessfully called saveStats" << endl;
  //cout << "\tFinished call to eval_class for class " << CLASS_NAMES[current_class] << endl;
  return true;
}


inline bool exists_test0(const std::string &name)
{
  return (access(name.c_str(), F_OK) != -1);
}

void saveAndPlotPlots(string dir_name, string file_name, string obj_type, vector<double> vals[], bool is_aos)
{
  char command[1024];

  // save plot data to file
  //cout << "\t SaveAndPlotPlots: Writing file" << dir_name + "/" + file_name + ".txt" << endl;
  FILE *fp = fopen((dir_name + "/" + file_name + ".txt").c_str(), "w");
  //cout << "Saving plot to file " << dir_name + "/" + file_name + ".txt" << endl;
  for (int32_t i = 0; i < (int)N_SAMPLE_PTS; i++)
    fprintf(fp, "%f %f %f %f\n", (double)i / (N_SAMPLE_PTS - 1.0), vals[0][i], vals[1][i], vals[2][i]);
  fclose(fp);

  // create png + eps
  for (int32_t j = 0; j < 2; j++)
  {

    // open file
    FILE *fp = fopen((dir_name + "/" + file_name + ".gp").c_str(), "w");

    // save gnuplot instructions
    if (j == 0)
    {
      fprintf(fp, "set term png size 450,315 font \"Helvetica\" 11\n");
      fprintf(fp, "set output \"%s.png\"\n", file_name.c_str());
    }
    else
    {
      fprintf(fp, "set term postscript eps enhanced color font \"Helvetica\" 20\n");
      fprintf(fp, "set output \"%s.eps\"\n", file_name.c_str());
    }

    // set labels and ranges
    fprintf(fp, "set size ratio 0.7\n");
    fprintf(fp, "set xrange [0:1]\n");
    fprintf(fp, "set yrange [0:1]\n");
    fprintf(fp, "set xlabel \"Recall\"\n");
    if (!is_aos)
      fprintf(fp, "set ylabel \"Precision\"\n");
    else
      fprintf(fp, "set ylabel \"Orientation Similarity\"\n");
    obj_type[0] = toupper(obj_type[0]);
    fprintf(fp, "set title \"%s\"\n", obj_type.c_str());

    // line width
    int32_t lw = 5;
    if (j == 0)
      lw = 3;

    // plot error curve
    fprintf(fp, "plot ");
    fprintf(fp, "\"%s.txt\" using 1:2 title 'Easy' with lines ls 1 lw %d,", file_name.c_str(), lw);
    fprintf(fp, "\"%s.txt\" using 1:3 title 'Moderate' with lines ls 2 lw %d,", file_name.c_str(), lw);
    fprintf(fp, "\"%s.txt\" using 1:4 title 'Hard' with lines ls 3 lw %d", file_name.c_str(), lw);

    // close file
    fclose(fp);

    // run gnuplot => create png + eps
    sprintf(command, "cd %s; gnuplot %s", dir_name.c_str(), (file_name + ".gp").c_str());
    system(command);
  }

  // create pdf and crop
  sprintf(command, "cd %s; ps2pdf %s.eps %s_large.pdf", dir_name.c_str(), file_name.c_str(), file_name.c_str());
  system(command);
  sprintf(command, "cd %s; pdfcrop %s_large.pdf %s.pdf", dir_name.c_str(), file_name.c_str(), file_name.c_str());
  system(command);
  sprintf(command, "cd %s; rm %s_large.pdf", dir_name.c_str(), file_name.c_str());
  system(command);
}

int32_t eval_class_and_plot(bool do_eval_class, CLASSES CLASS_TYPE, string result_dir, string plot_dir, vector<vector<tGroundtruth>> &groundtruth, vector<vector<tDetection>> &detections, bool compute_aos)
{
  /* Evaluates the given class if the bool eval_class is set, and saves the outcome of the evaluation from save class to
  plot and file */
  // eval traffic lights
  // holds pointers for result files
  FILE *fp_det = 0, *fp_ori = 0;
  //cout << "\tEntering eval_class_and_plot - class: " << CLASS_NAMES[CLASS_TYPE] << " do evaluation? : " << do_eval_class <<  endl;
  //cout << "\tSize of ground truths and detections " << groundtruth.size() << ", " << detections.size() << endl;
  if (do_eval_class){
    //cout << "\tEvaluating class " << CLASS_NAMES[CLASS_TYPE] << endl;
    //cout << "\tResult file will be in directory " << result_dir << endl;
   
    fp_det = fopen((result_dir + "/stats_" + CLASS_NAMES[CLASS_TYPE] + "_detection.txt").c_str(), "w");
    //cout << "\tSuccessfully opened file " << result_dir + "/stats_" + CLASS_NAMES[CLASS_TYPE] + "_detection.txt" << endl;
    if (compute_aos)
      fp_ori = fopen((result_dir + "/stats_" + CLASS_NAMES[CLASS_TYPE] + "_orientation.txt").c_str(), "w");
    vector<double> precision[3], aos[3];
    //cout << "\tAbout to call eval class for class " << CLASS_NAMES[CLASS_TYPE] << endl;
    if (!eval_class(fp_det, fp_ori, CLASS_TYPE, groundtruth, detections, compute_aos, precision[0], aos[0], EASY) ||
        !eval_class(fp_det, fp_ori, CLASS_TYPE, groundtruth, detections, compute_aos, precision[1], aos[1], MODERATE) ||
        !eval_class(fp_det, fp_ori, CLASS_TYPE, groundtruth, detections, compute_aos, precision[2], aos[2], HARD)){
      cout << "===============" <<  CLASS_NAMES[CLASS_TYPE] << " evaluation failed." << endl;
      return false;
    }else{
      cout << CLASS_NAMES[CLASS_TYPE] << " successfully evaluated" << endl;
    }
    fclose(fp_det);
    saveAndPlotPlots(plot_dir, CLASS_NAMES[CLASS_TYPE] + "_detection", CLASS_NAMES[CLASS_TYPE], precision, 0);
    //cout << "\tFinished saving plots with class type " << CLASS_NAMES[CLASS_TYPE] << endl;
    if (compute_aos)
    {
      fclose(fp_ori);
      saveAndPlotPlots(plot_dir, CLASS_NAMES[CLASS_TYPE] + "_orientation", CLASS_NAMES[CLASS_TYPE], aos, 1);
    }
  }else{
    cout << "Eval class and plot called, but class " << CLASS_NAMES[CLASS_TYPE] << " is set to skip. " << endl;
  }
}
#include <fstream>
#include <experimental/filesystem>

bool has_suffix(const std::string &str, const std::string &suffix){
  return str.size() >= suffix.size() &&
         str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void load_gt_and_det(string path, string result_dir, string gt_dir, vector<vector<tGroundtruth>> &groundtruth, vector<vector<tDetection>> &detections,  bool &eval_car, bool &eval_person, bool &eval_bike, bool &eval_trafficlight, bool &eval_trafficsign, bool &eval_truck, bool &compute_aos){
  // for all images read groundtruth and detections
  // Get all filenames from gt
  // This is necessary since KittiBox only uses numbers as filenames, but we have IDs
  namespace fs = std::experimental::filesystem;
  cout << "Loading detections..." << endl;
  // Iterate through all files, loading detections and groundtruths. 
  for (auto &file_path : fs::directory_iterator(path))
  {
    // file name
    string full_file_name = file_path.path().string();
    // Since the file names are given as absolute paths, we need to find the last folder separator and ignore everything before this.
    string file_name(full_file_name.substr(full_file_name.rfind("/") + 1));
    //cout << "\tload_gt_and_det: reading file " << file_name << endl;
    if (!exists_test0(result_dir + "/" + file_name) || !has_suffix(file_name, ".txt") || has_suffix(file_name, "orientation.txt") || has_suffix(file_name, "detection.txt"))
    {
      cout << "\t" << result_dir + "/" + file_name << " does not exist or does not contain detections - skipping" << endl;
      continue;
    }

    // read ground truth and result poses
    bool gt_success, det_success;
    //cout << "\t eval(): Trying to load ground truths from " << gt_dir + "/" + file_name << endl;
    vector<tGroundtruth> gt = loadGroundtruth(gt_dir + "/" + file_name, gt_success);
    //cout << "\t eval(): Trying to load detections from " << result_dir + "/" + file_name << endl;
    vector<tDetection> det = loadDetections(result_dir + "/" + file_name, compute_aos, eval_car, eval_person, eval_bike, eval_trafficlight, eval_trafficsign, eval_truck, det_success);
    groundtruth.push_back(gt);
    detections.push_back(det);

    // check for errors
    if (!gt_success){
      cout << "eval(): ERROR: Couldn't read:" << file_name << " of ground truth (loadGroundTruth)." << endl;
    }
    if (!det_success){
      cout << "eval(): ERROR: Couldn't read:" << file_name << " of detections (loadDetections) " << endl;
    }
  }
  cout << "Size of gts and detections after loading: " << groundtruth.size() << ", " << detections.size() << endl;
}


bool eval(string path, string path_to_gt){
  // set some global parameters
  cout << "Initializing global variables ... " << endl;
  initGlobals();

  // ground truth and result directories
  string gt_dir = path_to_gt;
  string result_dir = path;
  string plot_dir = result_dir + "/plot";

  // create output directories
  system(("mkdir " + plot_dir).c_str());
  cout << "Output directories creates (mkdir " << plot_dir << ")" << endl;
  // holds wether orientation similarity shall be computed (might be set to false while loading detections)
  // and which labels where provided by this submission
  bool compute_aos = false, eval_car = true, eval_person = true, eval_bike = true;
  bool eval_trafficlight = true, eval_trafficsign = true, eval_truck = true;
  // FIXME: This is super hacky, but since groundtruths and detections are modified in the code,
  // load them for evaluating each class.
  // hold detections and ground truth in memory
  vector<vector<tGroundtruth>> groundtruth_person;
  vector<vector<tDetection>> detections_person;
  load_gt_and_det(path, result_dir, gt_dir, groundtruth_person, detections_person, eval_car, eval_person, eval_bike, eval_trafficlight, eval_trafficsign, eval_truck, compute_aos);
  eval_class_and_plot(eval_person, PERSON, result_dir, plot_dir, groundtruth_person, detections_person, compute_aos);
  cout << "Finished evaluation person " << endl;

  vector<vector<tGroundtruth>> groundtruth_trafficsign;
  vector<vector<tDetection>> detections_trafficsign;
  load_gt_and_det(path, result_dir, gt_dir, groundtruth_trafficsign, detections_trafficsign, eval_car, eval_person, eval_bike, eval_trafficlight, eval_trafficsign, eval_truck, compute_aos);
  eval_class_and_plot(eval_trafficsign, TRAFFIC_SIGN, result_dir, plot_dir, groundtruth_trafficsign, detections_trafficsign, compute_aos);
  cout << "Finished evaluation traffic sign " << endl;
  
  vector<vector<tGroundtruth>> groundtruth_bike;
  vector<vector<tDetection>> detections_bike;
  load_gt_and_det(path, result_dir, gt_dir,groundtruth_bike, detections_bike, eval_car, eval_person, eval_bike, eval_trafficlight, eval_trafficsign, eval_truck, compute_aos);
  eval_class_and_plot(eval_bike, BIKE, result_dir, plot_dir, groundtruth_bike, detections_bike, compute_aos);
  cout << "Finished evaluation bike" << endl;
  
  vector<vector<tGroundtruth>> groundtruth_car;
  vector<vector<tDetection>> detections_car;
  load_gt_and_det(path, result_dir, gt_dir, groundtruth_car, detections_car, eval_car, eval_person, eval_bike, eval_trafficlight, eval_trafficsign, eval_truck, compute_aos);
  eval_class_and_plot(eval_car, CAR, result_dir, plot_dir, groundtruth_car, detections_car, compute_aos);
  
  vector<vector<tGroundtruth>> groundtruth_trafficlight;
  vector<vector<tDetection>> detections_trafficlight;
  load_gt_and_det(path, result_dir, gt_dir, groundtruth_trafficlight, detections_trafficlight, eval_car, eval_person, eval_bike, eval_trafficlight, eval_trafficsign, eval_truck, compute_aos);
  cout << "Finished evaluation car " << endl;
  eval_class_and_plot(eval_trafficlight, TRAFFIC_LIGHT, result_dir, plot_dir, groundtruth_trafficlight, detections_trafficlight, compute_aos);
  cout << "Finished evaluation traffic light " << endl;
  
  vector<vector<tGroundtruth>> groundtruth_truck;
  vector<vector<tDetection>> detections_truck;
  load_gt_and_det(path, result_dir, gt_dir, groundtruth_truck, detections_truck,  eval_car, eval_person, eval_bike, eval_trafficlight, eval_trafficsign, eval_truck, compute_aos);
  eval_class_and_plot(eval_truck, TRUCK, result_dir, plot_dir, groundtruth_truck, detections_truck, compute_aos);
  cout << "Finished evaluation truck" << endl;
  // success
  return true;
}

int32_t main(int32_t argc, char *argv[])
{

  // we need 2 or 4 arguments!
  if (argc != 3 && argc != 4)
  {
    cout << "Usage: ./eval_detection path_to_prediction path_to_ground_truth" << endl;
    cout << "ARGC: " << argc;
    return 1;
  }
  cout << "************************************************** ENTERING CPP EVAL CODE *******************";
  cout << "*********************************************************************************************";
  cout << "*********************************************************************************************" << endl;
  cerr << "TESTING TESTING TESTING TESTING CERR " << endl;
  cout << "Running main of cpp evaluation" << endl;
  // read arguments
  cout << "Reading arguments" << endl;
  string path_to_prediction = argv[1];
  string path_to_gt = argv[2];
  cout << "Path to prediction: " << path_to_prediction << endl;
  cout << "Path to ground truths: " << path_to_gt << endl;
  // init notification mail
  cout << "Starting to evaluate results - found in " << path_to_prediction.c_str() << endl;
  // run evaluation
  if (eval(path_to_prediction, path_to_gt))
  {
    cout << "CPP EVAL: Evaluation sucessful! " << endl;
    cout << "***************************************************** EXITING CPP EVAL CODE *******************" << endl;
    return 0;
  }
  else
  {
    cout << "CPP EVAL: An error occured while processing your results." << endl;
    cout << "***************************************************** EXITING CPP EVAL CODE *******************" << endl;
    return 1;
  }

  return 0;
}




