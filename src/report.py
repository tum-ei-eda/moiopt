import pathdiscovery as pd
import graph_analyzer
import memplanner
import moiopt


def viabilityReport(mod):
    analyzer = graph_analyzer.GraphAnalyzer()
    analyzer.run(mod["main"])
    sn = analyzer.makeNet()
    sched = sn.createBestSchedule()
    planner = memplanner.MemoryPlanner(sched)

    def getCritBufs(memLayout):
        return {buf: memLayout.getOffset(buf) for buf in memLayout.getBufsByCriticality()}

    critBufsWithOffsets = memplanner.memLayoutWithTimeout(sn, planner, getCritBufs)

    # Remove input and output buffers.
    inOutBufs = sn.getInBufs() + sn.getOutBufs()
    critBufsWithOffsets = {buf: offset for buf, offset in critBufsWithOffsets.items() if buf not in inOutBufs}

    print("Number of potential critical buffers:", len(critBufsWithOffsets))
    for buf, offset in critBufsWithOffsets.items():
        print("  POT Crit buf:", buf.size, buf.name)

    excludedBufs = []
    for buf in critBufsWithOffsets:
        targetExpr = analyzer.getExprFromBuf(buf)
        dmod, targetExpr = moiopt.defuseAndRememberExpr(mod, targetExpr)
        danalyzer = graph_analyzer.GraphAnalyzer()
        danalyzer.run(dmod["main"])
        dsn = danalyzer.makeNet()

        discovery = pd.PathDiscovery(targetExpr, danalyzer, dsn)
        discovery.discoverAll()
        if len(discovery.splitPaths) == 0:
            excludedBufs.append(buf)

    critBufsWithOffsets = {buf: offset for buf, offset in critBufsWithOffsets.items() if buf not in excludedBufs}


    print("Number of critical buffers:", len(critBufsWithOffsets))
    for buf, offset in critBufsWithOffsets.items():
        print("  Crit buf:", buf.size, buf.name)

    #targetExpr = analyzer.getExprFromBuf(maxOp.getOutputs()[0])
    #discovery = pd.PathDiscovery(targetExpr, analyzer, sn)
