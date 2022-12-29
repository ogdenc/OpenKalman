/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definitions for Eigen3::EigenWrapper
 */

#ifndef OPENKALMAN_EIGENWRAPPER_HPP
#define OPENKALMAN_EIGENWRAPPER_HPP

namespace OpenKalman
{
  namespace Eigen3
  {
    template<typename T>
    struct EigenWrapper : T {};
  } // namespace Eigen3


  // ------------ //
  //  Interfaces  //
  // ------------ //

  namespace interface
  {
    using namespace OpenKalman::Eigen3;

    template<typename T>
    struct IndexibleObjectTraits<EigenWrapper<T>> : IndexibleObjectTraits<T> {};

    template<typename T, std::size_t N>
    struct IndexTraits<EigenWrapper<T>, N> : IndexTraits<T, N> {};

    template<typename T, std::size_t N>
    struct CoordinateSystemTraits<EigenWrapper<T>, N> : CoordinateSystemTraits<T, N> {};

    template<typename T, typename...I>
#ifdef __cpp_concepts
    struct GetElement<EigenWrapper<T>, I...>
#else
    struct GetElement<EigenWrapper<T>, void, I...>
#endif
      : GetElement<T, I...> {};

    template<typename T, typename...I>
#ifdef __cpp_concepts
    struct SetElement<EigenWrapper<T>, I...>
#else
    struct SetElement<EigenWrapper<T>, void, I...>
#endif
      : SetElement<T, I...> {};

    template<typename T, typename Scalar>
    struct EquivalentDenseWritableMatrix<EigenWrapper<T>, Scalar> : EquivalentDenseWritableMatrix<std::decay_t<T>, Scalar>
    {
      template<typename Arg>
      static decltype(auto) convert(Arg&& arg)
      {
        using M = Eigen::Matrix<Scalar, 1, 1>;
        return EquivalentDenseWritableMatrix<M, Scalar>::convert(std::forward<Arg>(arg));
      }

      template<typename Arg>
      static decltype(auto) to_native_matrix(Arg&& arg) { return std::forward<Arg>(arg); }
    };

    template<typename T>
    struct Dependencies<EigenWrapper<T>> : Dependencies<T> {};

    template<typename T, typename Scalar>
    struct SingleConstantMatrixTraits<EigenWrapper<T>, Scalar> : SingleConstantMatrixTraits<T, Scalar> {};

    template<typename T>
    struct SingleConstant<EigenWrapper<T>> : SingleConstant<T> {};

    template<typename T>
    struct SingleConstantDiagonalMatrixTraits<EigenWrapper<T>> : SingleConstantDiagonalMatrixTraits<std::decay_t<T>> {};

    template<typename T>
    struct SingleConstantDiagonal<EigenWrapper<T>> : SingleConstantDiagonal<T> {};

    template<typename T>
    struct DiagonalTraits<EigenWrapper<T>> : DiagonalTraits<T> {};

    template<typename T>
    struct TriangularTraits<EigenWrapper<T>> : TriangularTraits<T> {};

    template<typename T>
    struct HermitianTraits<EigenWrapper<T>> : HermitianTraits<T> {};

    template<typename T>
    struct ArrayOperations<EigenWrapper<T>> : ArrayOperations<T> {};

    template<typename T>
    struct Conversions<EigenWrapper<T>> : Conversions<T> {};

    template<typename T>
    struct Subsets<EigenWrapper<T>> : Subsets<T> {};

    template<typename T>
    struct ModularTransformationTraits<EigenWrapper<T>> : ModularTransformationTraits<T> {};

    template<typename T>
    struct LinearAlgebra<EigenWrapper<T>> : LinearAlgebra<T> {};

  } // namespace interface


  template<typename T>
  struct MatrixTraits<Eigen3::EigenWrapper<T>> : MatrixTraits<std::decay_t<T>> {};

} // namespace OpenKalman

#endif //OPENKALMAN_EIGENWRAPPER_HPP
