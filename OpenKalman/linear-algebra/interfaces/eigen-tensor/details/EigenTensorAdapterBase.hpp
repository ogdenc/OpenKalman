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
 * \brief Definitions for Eigen3::EigenTensorAdapterBase
 */

#ifndef OPENKALMAN_EIGENTENSORADAPTERBASE_HPP
#define OPENKALMAN_EIGENTENSORADAPTERBASE_HPP

namespace OpenKalman::Eigen3
{
  template<typename Derived, typename NestedMatrix>
  struct EigenTensorAdapterBase : EigenCustomBase,
    Eigen::TensorBase<Derived, Eigen::internal::accessors_level<std::decay_t<NestedMatrix>>::value>
  {

  private:

    using Base = Eigen::TensorBase<Derived, Eigen::internal::accessors_level<std::decay_t<NestedMatrix>>::value>;

  public:

    EigenTensorAdapterBase() = default;

    EigenTensorAdapterBase(const EigenTensorAdapterBase&) = default;

    EigenTensorAdapterBase(EigenTensorAdapterBase&&) = default;

    ~EigenTensorAdapterBase() = default;

    using Base::operator=;

    using typename Base::Scalar;

    /* \internal
     * \brief The underlying numeric type for composed scalar types.
     * \details In cases where Scalar is e.g. std::complex<T>, T were corresponding to RealScalar.
     */
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;


    /**
     * \internal
     * \brief The type of *this that is used for nesting within other Eigen classes.
     * \note Eigen3 requires this as the type used when Derived is nested.
     */
    using Nested = Derived;

    using StorageKind [[maybe_unused]] = typename Eigen::internal::traits<Derived>::StorageKind;

  };

}


#endif
