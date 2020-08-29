/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGENCOVARIANCEBASE_H
#define OPENKALMAN_EIGENCOVARIANCEBASE_H

namespace Eigen
{
  template<typename CovarianceType>
  struct CovarianceCommaInitializer;
}

namespace OpenKalman::internal
{
  /**
   * Ultimate base of Covariance and SquareRootCovariance classes, general case.
   * No conversion is necessary if either
   * (1) Derived is not a square root and the base is self-adjoint; or
   * (2) Derived is a square root and the base is triangular.
   */
  template<typename Derived, typename Nested>
  struct EigenCovarianceBase<Derived, Nested,
    std::enable_if_t<(is_self_adjoint_v<Nested> and not is_square_root_v<Derived>) or
      (is_triangular_v<Nested> and is_square_root_v<Derived>)>>
    : EigenMatrixBase<Derived, Nested> {};


  /**
   * Ultimate base of Covariance and SquareRootCovariance classes, if Derived is not a square root and
   * the base is not self-adjoint (i.e., it is triangular but not diagonal).
   */
  template<typename Derived, typename ArgType>
  struct EigenCovarianceBase<Derived, ArgType,
    std::enable_if_t<not is_self_adjoint_v<ArgType> and not is_square_root_v<Derived>>>
    : EigenMatrixBase<Derived, ArgType>
  {
    using Nested = std::decay_t<ArgType>;
    using Scalar = typename Nested::Scalar;
    using Base = EigenMatrixBase<Derived, Nested>;

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    constexpr auto operator<<(const S& s)
    {
      auto& xpr = static_cast<Derived&>(*this);
      return Eigen::CovarianceCommaInitializer(xpr, static_cast<const Scalar&>(s));
    }

    template<typename OtherDerived>
    constexpr auto operator<<(const Eigen::DenseBase<OtherDerived>& other)
    {
      auto& xpr = static_cast<Derived&>(*this);
      return Eigen::CovarianceCommaInitializer(xpr, other);
    }
  };


  /**
   * Ultimate base of Covariance and SquareRootCovariance classes, if Derived is a square root and
   * the base is not triangular (i.e., it is self-adjoint but not diagonal).
   */
  template<typename Derived, typename ArgType>
  struct EigenCovarianceBase<Derived, ArgType,
    std::enable_if_t<not is_triangular_v<ArgType> and is_square_root_v<Derived>>>
    : EigenMatrixBase<Derived, ArgType>
  {
    using Nested = std::decay_t<ArgType>;
    using Scalar = typename Nested::Scalar;
    using Base = EigenMatrixBase<Derived, Nested>;

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    constexpr auto operator<<(const S& s)
    {
      auto& xpr = static_cast<Derived&>(*this);
      return Eigen::CovarianceCommaInitializer(xpr, static_cast<const Scalar&>(s));
    }

    template<typename OtherDerived>
    constexpr auto operator<<(const Eigen::DenseBase<OtherDerived>& other)
    {
      auto& xpr = static_cast<Derived&>(*this);
      return Eigen::CovarianceCommaInitializer(xpr, other);
    }
  };

}


namespace Eigen
{
  /**
   * Alternative version of CommaInitializer for Covariance and SquareRootCovariance.
   */
  template<typename CovarianceType>
  struct CovarianceCommaInitializer
  {
    using Scalar = typename CovarianceType::Scalar;
    using BaseMatrix = OpenKalman::strict_matrix_t<typename OpenKalman::MatrixTraits<CovarianceType>::BaseMatrix>;
    using Nested = CommaInitializer<BaseMatrix>;

    BaseMatrix matrix;
    Nested comma_initializer;
    CovarianceType& cov;

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    CovarianceCommaInitializer(CovarianceType& xpr, const S& s)
      : matrix(), comma_initializer(matrix, static_cast<const Scalar&>(s)), cov(xpr) {}

    template<typename OtherDerived>
    CovarianceCommaInitializer(CovarianceType& xpr, const DenseBase <OtherDerived>& other)
      : matrix(), comma_initializer(matrix, other), cov(xpr) {}

    CovarianceCommaInitializer(const CovarianceCommaInitializer& o)
      : matrix(o.matrix), comma_initializer(o.comma_initializer), cov(o.cov) {}

    CovarianceCommaInitializer(CovarianceCommaInitializer&& o)
      : matrix(std::move(o.matrix)),
        comma_initializer(std::move(o.comma_initializer)), cov(std::move(o.cov)) {}

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator,(const S& s)
    {
      comma_initializer, static_cast<const Scalar&>(s);
      return *this;
    }

    template<typename OtherDerived>
    auto& operator,(const DenseBase<OtherDerived>& other)
    {
      comma_initializer, other;
      return *this;
    }

    ~CovarianceCommaInitializer()
    {
      finished();
    }

    auto& finished()
    {
      cov = comma_initializer.finished();
      return cov;
    }
  };

} // end namespace Eigen


#endif //OPENKALMAN_EIGENCOVARIANCEBASE_H
