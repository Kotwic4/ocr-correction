import logging
import re

from ocr_correction.dataset.icdar.evalTool_ICDAR2017 import EvalContext, maxNbCandidate
from ocr_correction.solutions.solution import Solution


class ICDAREval(EvalContext):
    def __init__(self, file_path):
        super(ICDAREval, self).__init__(file_path)
        self.tokensPos = [0] + [spacePos.start() + 1 for spacePos in re.finditer(r" ", self.ocrOriginal)]
        self.token_pos_map = {v: i for i, v in enumerate(self.tokensPos)}
        self.tokens = self.ocrOriginal.split(' ')
        self.file_path = file_path

    def collect_difference(self, show_errors=True, show_ignore=True):
        errorList = []

        gsAlignedHyphensIgnored = self.gsAligned
        for hToken in re.finditer(r"[^ ]*((\ ?-[^ ])|([^ ]-\ ?))[^ ]*", self.ocrAligned):
            gsAlignedHyphensIgnored = gsAlignedHyphensIgnored[:hToken.start()] + self.charIgnore * (
                    hToken.end() - hToken.start()) + gsAlignedHyphensIgnored[
                                                     hToken.end():]

        gsSpacePos = set([spacePos.start() for spacePos in re.finditer(r"^|$|\ ", gsAlignedHyphensIgnored)])
        ocrSpacePos = set([spacePos.start() for spacePos in re.finditer(r"^|$|\ ", self.ocrAligned)])
        commonSpacePos = sorted(gsSpacePos.intersection(ocrSpacePos))

        for i in range(len(commonSpacePos) - 1):

            tokenStartPos = commonSpacePos[i] + 1 * (gsAlignedHyphensIgnored[commonSpacePos[i]] == " ")
            tokenEndPos = commonSpacePos[i + 1]

            tokenInGs = re.sub(self.charExtend, "", gsAlignedHyphensIgnored[tokenStartPos:tokenEndPos])
            tokenInOcr = re.sub(self.charExtend, "", self.ocrAligned[tokenStartPos:tokenEndPos])

            # Get not aligned pos
            tokenStartPosOriginal = tokenStartPos - self.get_original_shift(tokenStartPos)

            ingored_char = self.charIgnore in tokenInGs

            if tokenInOcr != tokenInGs and (ingored_char and show_ignore or not ingored_char and show_errors):
                num_token = tokenInOcr.count(" ") + 1
                start_i = self.token_pos_map[tokenStartPosOriginal]
                end_i = start_i + num_token

                errorList.append((start_i, end_i, tokenInOcr, tokenInGs, ingored_char))

        return errorList

    def collect_errors(self):
        return self.collect_difference(show_errors=True, show_ignore=False)

    def collect_not_important(self):
        return self.collect_difference(show_errors=False, show_ignore=True)

    def parse_token_errors(self, errors_range):
        result_map = {}
        for start_i, end_i, candidates in errors_range:
            start_pos = self.tokensPos[start_i]
            end_i = end_i - start_i + 1
            result_map[f"{start_pos}:{end_i}"] = candidates

        tokenPosErrReshaped = {}
        for pos_nbtokens, candidates in result_map.items():

            pos, nbOverlappingToken = [int(i) for i in pos_nbtokens.split(':')]
            boundsAligned = self.get_aligned_token_bounds(pos, nbOverlappingToken)

            # Check pos targets a existing token (first char)
            assert pos in self.tokensPos, \
                "[ERROR] : Error at pos %s does not target the first char of a token (space separated sequences)." % pos

            assert self.ocrOriginal[pos:].count(" ") >= nbOverlappingToken - 1, \
                "[ERROR] : Error at pos %d spreads overs %d tokens which goes ouside the sequence." % (
                    pos, nbOverlappingToken)

            # Normalize candidates weights if needed
            normCandidates = {}

            # Limit the number of candiates
            for k, v in sorted(candidates.items(), key=lambda kv: kv[1], reverse=True):
                normCandidates[k] = v
                if len(normCandidates) >= maxNbCandidate:
                    break

            if len(normCandidates) > 0 and sum(normCandidates.values()) != 1:
                print("[WARNING] : Normalizing weights at %s:%s" % (pos_nbtokens, str(normCandidates)))
                normCandidates = {cor: float(x) / sum(normCandidates.values()) for cor, x in normCandidates.items()}

            tokenPosErrReshaped[pos] = {
                "nbToken": nbOverlappingToken,
                "boundsAligned": boundsAligned,
                "ocrSeq": re.sub(self.charExtend, "",
                                 self.ocrAligned[boundsAligned[0]:boundsAligned[0] + boundsAligned[1]]),
                "gsSeq": re.sub(self.charExtend, "",
                                self.gsAligned[boundsAligned[0]:boundsAligned[0] + boundsAligned[1]]),
                "candidates": normCandidates
            }
        return tokenPosErrReshaped

    def calculate_solution_stats(self, fixed_errors):
        tokenPosErrReshaped = self.parse_token_errors(fixed_errors)
        logging.debug(tokenPosErrReshaped)

        # Task 1) Run the evaluation : Detection of the position of erroneous tokens
        prec, recall, fmes = self.task1_eval(tokenPosErrReshaped)
        logging.debug((prec, recall, fmes))

        # Task 2) Run the evaluation : Correction of the erroneous tokens
        sumCorrectedDistance, sumOriginalDistance, nbSymbolsConsidered = self.task2_eval(tokenPosErrReshaped,
                                                                                         useFirstCandidateOnly=False)
        logging.debug((sumCorrectedDistance, sumOriginalDistance, nbSymbolsConsidered))

        # Manage division per zero
        avgCorrectedDistance = sumCorrectedDistance / float(nbSymbolsConsidered) if nbSymbolsConsidered > 0 else 0
        avgOriginalDistance = sumOriginalDistance / float(nbSymbolsConsidered) if nbSymbolsConsidered > 0 else 0

        if avgOriginalDistance > 0:
            improvement = (avgOriginalDistance - avgCorrectedDistance) / avgOriginalDistance
        else:
            improvement = 0

        nbTokens, nbErrTokens, nbErrTokensAlpha = self.get_errors_stats()
        logging.debug((nbTokens, nbErrTokens, nbErrTokensAlpha))

        return (
            self.file_path,
            nbTokens,
            nbErrTokens,
            nbSymbolsConsidered,
            prec,
            recall,
            fmes,
            avgOriginalDistance,
            avgCorrectedDistance,
            improvement
        )

    def check_solution(self, solution: Solution):

        # TODO calculate time
        logging.info(self.file_path)

        tokens = self.tokens
        logging.debug(tokens)

        e_tokens = solution.find_errors(tokens)
        logging.debug(e_tokens)

        fixed_errors = solution.fix_errors(tokens, e_tokens)
        logging.debug(fixed_errors)

        result = self.calculate_solution_stats(fixed_errors)
        logging.info(result)

        return result
