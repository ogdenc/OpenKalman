/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Native Eigen evaluator for \ref internal::LibraryWrapper "LibraryWrapper".
 */

#ifndef OPENKALMAN_EIGEN_NATIVE_EVALUATORS_LIBRARYWRAPPER_HPP
#define OPENKALMAN_EIGEN_NATIVE_EVALUATORS_LIBRARYWRAPPER_HPP


namespace OpenKalman::detail
{

#ifdef __cpp_concepts
  template<typename XprType, typename Nested>
#else
  template<typename XprType, typename Nested, typename = void>
#endif
  struct EigenWrapperEvaluatorBase : Eigen::internal::evaluator_base<XprType>
  {
    explicit EigenWrapperEvaluatorBase(const XprType& t) : m_xpr {const_cast<XprType&>(t)} {}


    auto& coeffRef(Eigen::Index row, Eigen::Index col)
    {
      return OpenKalman::get_component(m_xpr.nested_object(), static_cast<std::size_t>(row), static_cast<std::size_t>(col));
    }


    constexpr decltype(auto) coeff(Eigen::Index row, Eigen::Index col) const
    {
      return OpenKalman::get_component(m_xpr.nested_object(), static_cast<std::size_t>(row), static_cast<std::size_t>(col));
    }


    enum {
      CoeffReadCost = 0,
      Flags = Eigen::internal::traits<XprType>::Flags,
      Alignment = Eigen::AlignedMax
    };

  protected:

    XprType& m_xpr;
  };


#ifdef __cpp_concepts
  template<typename XprType, OpenKalman::Eigen3::eigen_dense_general Nested> requires
    requires { typename Eigen::internal::evaluator<std::decay_t<Nested>>; }
  struct EigenWrapperEvaluatorBase<XprType, Nested>
#else
  template<typename XprType, typename Nested>
  struct EigenWrapperEvaluatorBase<XprType, Nested, std::enable_if_t<OpenKalman::Eigen3::eigen_dense_general<Nested>>>
#endif
    : Eigen::internal::evaluator<std::decay_t<Nested>>
  {
    explicit EigenWrapperEvaluatorBase(const XprType& t)
      : Eigen::internal::evaluator<std::decay_t<Nested>> {t.nested_object()} {}
  };

} // namespace OpenKalman::detail


namespace Eigen::internal
{

  template<typename T, typename L>
  struct evaluator<OpenKalman::internal::LibraryWrapper<T, L>>
    : OpenKalman::detail::EigenWrapperEvaluatorBase<OpenKalman::internal::LibraryWrapper<T, L>, T>
  {
    static_assert(OpenKalman::Eigen3::eigen_general<L>);
    using XprType = OpenKalman::internal::LibraryWrapper<T, L>;
    using Base = OpenKalman::detail::EigenWrapperEvaluatorBase<XprType, T>;
    explicit evaluator(const XprType& t) : Base {t} {}
  };

} // Eigen::internal

#endif //OPENKALMAN_EIGEN_NATIVE_EVALUATORS_LIBRARYWRAPPER_HPP