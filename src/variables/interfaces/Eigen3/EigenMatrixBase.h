/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGENMATRIXBASE_H
#define OPENKALMAN_EIGENMATRIXBASE_H

namespace Eigen
{
  template<typename Derived, typename XprType>
  struct MeanCommaInitializer;
}

namespace OpenKalman::internal
{
  /*
   * Base class for all OpenKalman classes that are also Eigen3 matrices.
   */
  template<typename Derived, typename ArgType>
  struct EigenMatrixBase : Eigen::MatrixBase<Derived>
  {
    using Nested = std::decay_t<ArgType>; ///< The nested Eigen matrix type. Eigen3 requires this to be defined.
    using Scalar = typename Nested::Scalar;

  protected:
    template<typename Arg>
    constexpr decltype(auto) get_ultimate_base_matrix(Arg&& arg) noexcept
    {
      decltype(auto) b = base_matrix(std::forward<Arg>(arg));
      using B = decltype(b);
      if constexpr(
        OpenKalman::is_EigenSelfAdjointMatrix_v<B> or
        OpenKalman::is_EigenTriangularMatrix_v<B> or
        OpenKalman::is_EigenDiagonal_v<B> or
        OpenKalman::is_FromEuclideanExpr_v<B> or
        OpenKalman::is_ToEuclideanExpr_v<B>)
      {
        return get_ultimate_base_matrix(b);
      }
      else
      {
        return b;
      }
    }

  public:

    static constexpr Eigen::Index rows() { return Eigen::internal::traits<Derived>::RowsAtCompileTime; } ///< Required by Eigen::EigenBase.

    static constexpr Eigen::Index cols() { return Eigen::internal::traits<Derived>::ColsAtCompileTime; } ///< Required by Eigen::EigenBase.

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    constexpr auto operator<<(const S& s)
    {
      auto& xpr = get_ultimate_base_matrix(static_cast<Derived&>(*this));
      using Xpr = std::decay_t<decltype(xpr)>;
      if constexpr(is_mean_v<Derived>)
        return Eigen::MeanCommaInitializer<Derived, Xpr>(xpr, static_cast<const Scalar&>(s));
      else
        return Eigen::CommaInitializer(xpr, static_cast<const Scalar&>(s));
    }

    template<typename OtherDerived>
    constexpr auto operator<<(const Eigen::DenseBase<OtherDerived>& other)
    {
      auto& xpr = get_ultimate_base_matrix(static_cast<Derived&>(*this));
      using Xpr = std::decay_t<decltype(xpr)>;
      if constexpr(is_mean_v<Derived>)
        return Eigen::MeanCommaInitializer<Derived, Xpr>(xpr, static_cast<const OtherDerived&>(other));
      else
        return Eigen::CommaInitializer(xpr, static_cast<const OtherDerived&>(other));
    }

    /// Refined to avoid confusion with zero().
    static decltype(auto) Zero() { return Derived::zero(); }

    /// Redefined to avoid confusion with identity().
    static decltype(auto) Identity() { return Derived::identity(); }
  };

}

namespace Eigen
{
  template<typename Derived, typename XprType>
  struct MeanCommaInitializer : CommaInitializer<XprType>
  {
    using Base = CommaInitializer<XprType>;
    using Scalar = typename XprType::Scalar;
    using Coefficients = typename OpenKalman::MatrixTraits<Derived>::RowCoefficients;
    using Base::Base;

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    MeanCommaInitializer(XprType& xpr, const S& s) : Base(xpr, static_cast<const Scalar&>(s)) {}

    template<typename OtherDerived>
    MeanCommaInitializer(XprType& xpr, const DenseBase <OtherDerived>& other) : Base(xpr, other) {}

    ~MeanCommaInitializer()
    {
      finished();
    }

    auto& finished()
    {
      this->m_xpr = OpenKalman::wrap_angles<Coefficients>(Base::finished());
      return this->m_xpr;
    }
  };

}

#endif //OPENKALMAN_EIGENMATRIXBASE_H
