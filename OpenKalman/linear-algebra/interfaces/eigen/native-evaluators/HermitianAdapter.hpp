/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Native Eigen3 evaluator for HermitianAdapter
 */

#ifndef OPENKALMAN_EIGEN_NATIVE_EVALUATORS_HERMITIANADAPTER_HPP
#define OPENKALMAN_EIGEN_NATIVE_EVALUATORS_HERMITIANADAPTER_HPP

#include <complex>

namespace Eigen::internal
{
  template<typename ArgType, OpenKalman::HermitianAdapterType storage_triangle>
  struct evaluator<OpenKalman::HermitianAdapter<ArgType, storage_triangle>>
    : evaluator_base<OpenKalman::HermitianAdapter<ArgType, storage_triangle>>
  {
    using XprType = OpenKalman::HermitianAdapter<ArgType, storage_triangle>;
    using NestedEvaluator = evaluator<std::decay_t<ArgType>>;
    using Scalar = typename traits<std::decay_t<ArgType>>::Scalar;
    using CoeffReturnType = typename std::decay_t<ArgType>::CoeffReturnType;

    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost,
      Flags = traits<XprType>::Flags,
      Alignment = NestedEvaluator::Alignment
    };


    explicit evaluator(const XprType& m_arg) : m_argImpl(m_arg.nested_object()) {}


    auto& coeffRef(Index row, Index col)
    {
      static_assert(not OpenKalman::value::complex<Scalar>,
        "Reference to element is not available for a complex HermitianAdapter");

      if constexpr (storage_triangle == OpenKalman::HermitianAdapterType::upper)
      {
        if (row > col)
          return m_argImpl.coeffRef(col, row);
      }
      else if constexpr (storage_triangle == OpenKalman::HermitianAdapterType::lower)
      {
        if (row < col)
          return m_argImpl.coeffRef(col, row);
      }

      return m_argImpl.coeffRef(row, col);
    }


    auto& coeffRef(Index i)
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Linear (single index) element access by reference is only available for one-by-one HermitianAdapter");

      static_assert(not OpenKalman::value::complex<Scalar>,
        "Reference to element is not available for a complex HermitianAdapter");

      return m_argImpl.coeffRef(i);
    }


    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr (storage_triangle == OpenKalman::HermitianAdapterType::upper)
      {
        if (row > col)
        {
          using std::conj;
          if constexpr (OpenKalman::value::complex<Scalar>) return conj(m_argImpl.coeff(col, row));
          else return m_argImpl.coeff(col, row);
        }
      }
      else if constexpr (storage_triangle == OpenKalman::HermitianAdapterType::lower)
      {
        if (row < col)
        {
          using std::conj;
          if constexpr (OpenKalman::value::complex<Scalar>) return conj(m_argImpl.coeff(col, row));
          else return m_argImpl.coeff(col, row);
        }
      }

      if (row == col)
      {
        if constexpr (OpenKalman::value::complex<Scalar> and not std::is_lvalue_reference_v<CoeffReturnType>)
        {
          using std::real;
          return real(m_argImpl.coeff(row, col));
        }
      }

      return m_argImpl.coeff(row, col);
    }


    CoeffReturnType coeff(Index i) const
    {
      return m_argImpl.coeff(i);
    }


    template<int LoadMode, typename PacketType>
    constexpr PacketType packet(Index row, Index col) const
    {
      return m_argImpl.template packet<LoadMode, PacketType>(row, col);
    }


    template<int LoadMode, typename PacketType>
    constexpr PacketType packet(Index index) const
    {
      return m_argImpl.template packet<LoadMode, PacketType>(index);
    }

  protected:

    NestedEvaluator m_argImpl;
  };


} // namespace Eigen::internal

#endif //OPENKALMAN_EIGEN_NATIVE_EVALUATORS_HERMITIANADAPTER_HPP
